"""
Knowledge Distillation Training Script for Hunyuan-MT-7B
Specialized for Chinese-Japanese Translation using WCC-JC 2.0 dataset
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from tqdm import tqdm
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ParallelTranslationDataset(Dataset):
    """Dataset for parallel Japanese-Chinese translation pairs"""
    
    def __init__(self, ja_file, zh_file, tokenizer, max_length=512, direction='ja2zh', preprocess=True):
        """
        Args:
            ja_file: Path to Japanese text file
            zh_file: Path to Chinese text file
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            direction: 'ja2zh' for Japanese->Chinese, 'zh2ja' for Chinese->Japanese
            preprocess: If True, pre-tokenize all data (faster but uses more memory)
        """
        with open(ja_file, 'r', encoding='utf-8') as f:
            self.ja_lines = [line.strip() for line in f.readlines()]
        
        with open(zh_file, 'r', encoding='utf-8') as f:
            self.zh_lines = [line.strip() for line in f.readlines()]
        
        assert len(self.ja_lines) == len(self.zh_lines), \
            f"Mismatch: {len(self.ja_lines)} Japanese lines vs {len(self.zh_lines)} Chinese lines"
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.direction = direction
        self.preprocess = preprocess
        
        # Pre-tokenize all data for faster training
        if preprocess:
            logger.info(f"Pre-tokenizing {len(self.ja_lines)} parallel sentences...")
            self.cached_data = []
            for idx in tqdm(range(len(self.ja_lines)), desc="Preprocessing"):
                item = self._tokenize_item(idx)
                self.cached_data.append(item)
            logger.info("Pre-tokenization complete!")
        else:
            self.cached_data = None
            logger.info(f"Loaded {len(self.ja_lines)} parallel sentences (will tokenize on-the-fly)")
    
    def __len__(self):
        return len(self.ja_lines)
    
    def _tokenize_item(self, idx):
        """Tokenize a single item (used for both preprocessing and on-the-fly)"""
        ja_text = self.ja_lines[idx]
        zh_text = self.zh_lines[idx]
        
        # Determine source and target based on direction
        if self.direction == 'ja2zh':
            source_text = ja_text
            target_text = zh_text
            target_language = "Chinese"
        else:  # zh2ja
            source_text = zh_text
            target_text = ja_text
            target_language = "Japanese"
        
        # Format as translation task using chat template
        messages = [
            {"role": "user", "content": f"Translate the following segment into {target_language}, use a casual tone, without additional explanation.\n\n{source_text}"}
        ]
        
        # Apply chat template and tokenize input
        tokenized_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        ).squeeze(0)
        
        # Tokenize target text
        target_messages = [
            {"role": "user", "content": f"Translate the following segment into {target_language}, use a casual tone, without additional explanation.\n\n{source_text}"},
            {"role": "assistant", "content": target_text}
        ]
        
        tokenized_full = self.tokenizer.apply_chat_template(
            target_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        ).squeeze(0)
        
        # Extract labels (shift by 1 for next token prediction)
        input_ids = tokenized_full[:-1]
        labels = tokenized_full[1:].clone()
        
        # Find where the assistant response starts (after <|extra_0|>)
        # We only compute loss on the target translation part
        try:
            extra_0_id = self.tokenizer.convert_tokens_to_ids("<|extra_0|>")
            if extra_0_id is not None and extra_0_id in input_ids:
                start_indices = (input_ids == extra_0_id).nonzero(as_tuple=True)[0]
                if len(start_indices) > 0:
                    # Mask everything before assistant response (keep the token after <|extra_0|>)
                    labels[:start_indices[-1] + 1] = -100
        except Exception as e:
            logger.debug(f"Could not find extra_0 token: {e}")
        
        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Truncate if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        
        # Pad if needed
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            padding = torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])
            
            label_padding = torch.full((padding_length,), -100, dtype=torch.long)
            labels = torch.cat([labels, label_padding])
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'source_text': source_text,
            'target_text': target_text
        }
    
    def __getitem__(self, idx):
        if self.cached_data is not None:
            return self.cached_data[idx]
        else:
            return self._tokenize_item(idx)


class DistillationTrainer:
    """Trainer for knowledge distillation"""
    
    def __init__(
        self,
        teacher_model,
        student_model,
        tokenizer,
        train_dataset,
        val_dataset=None,
        temperature=4.0,
        alpha=0.7,
        device='cuda',
        output_dir='./distilled_model',
        batch_size=4,
        learning_rate=5e-5,
        num_epochs=3,
        gradient_accumulation_steps=1,
        warmup_steps=100,
        max_length=512,
        save_steps=500,
        eval_steps=500,
        logging_steps=50,
        fp16=True
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.max_length = max_length
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.fp16 = fp16
        
        # Move models to device
        self.teacher_model.to(device)
        self.teacher_model.eval()  # Teacher is always in eval mode
        # Freeze teacher model completely for better performance
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        self.student_model.to(device)
        self.student_model.train()
        
        # Enable optimizations for better performance
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Use faster matmul precision on modern GPUs (Ampere+)
            try:
                torch.set_float32_matmul_precision('high')  # or 'medium' for better compatibility
            except AttributeError:
                pass  # Not available in older PyTorch versions
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            fused=True if device.type == 'cuda' else False  # Use fused optimizer on CUDA
        )
        
        # Calculate total training steps
        num_training_steps = len(train_dataset) // (batch_size * gradient_accumulation_steps) * num_epochs
        
        # Setup learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Setup scaler for mixed precision
        # Use BF16 for better stability and performance on modern GPUs
        self.use_amp = fp16 and device.type == 'cuda'
        self.amp_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.use_amp and self.amp_dtype == torch.float16))
        
        logger.info(f"Initialized DistillationTrainer")
        logger.info(f"  Total training steps: {num_training_steps}")
        logger.info(f"  Temperature: {temperature}, Alpha: {alpha}")
        logger.info(f"  Mixed precision: {self.use_amp} ({self.amp_dtype})")
    
    def compute_distillation_loss(self, student_logits, teacher_logits, labels):
        """Compute knowledge distillation loss (optimized version)"""
        # Pre-compute temperature scaling
        temp = self.temperature
        temp_sq = temp * temp
        
        # Soft targets: KL divergence between student and teacher
        # Use in-place operations where possible for better memory usage
        student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
        
        # Reshape once for efficiency
        vocab_size = student_log_probs.size(-1)
        student_log_probs_flat = student_log_probs.view(-1, vocab_size)
        teacher_probs_flat = teacher_probs.view(-1, vocab_size)
        
        # Compute KL divergence
        kl_loss = F.kl_div(
            student_log_probs_flat,
            teacher_probs_flat,
            reduction='batchmean',
            log_target=False
        ) * temp_sq
        
        # Hard targets: Cross-entropy with ground truth
        ce_loss = F.cross_entropy(
            student_logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        
        # Combined loss
        total_loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        
        return total_loss, kl_loss.detach(), ce_loss.detach()

    def get_teacher_logits(self, input_ids):
        """Get logits from teacher model (optimized)"""
        # Use inference_mode for faster inference (slightly faster than no_grad)
        with torch.inference_mode():
            # Move input to teacher device efficiently
            teacher_input = input_ids.to(self.teacher_model.device, non_blocking=True)
            
            # Use autocast for consistent dtype handling
            with torch.cuda.amp.autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.teacher_model(
                    input_ids=teacher_input,
                    labels=None,
                    output_hidden_states=False,
                    use_cache=False  # Disable cache for faster inference
                )
            
            # Extract logits (full sequence)
            logits = outputs.logits
            
            return logits
    
    def train_step(self, batch):
        """Perform a single training step (optimized)"""
        # Use non_blocking for faster transfer
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        labels = batch['labels'].to(self.device, non_blocking=True)
        
        # Use autocast for mixed precision training
        with torch.cuda.amp.autocast(dtype=self.amp_dtype, enabled=self.use_amp):
            # Forward pass through student (compute logits only, not loss)
            student_outputs = self.student_model(
                input_ids=input_ids,
                output_hidden_states=False,
                return_dict=True
            )
            
            # Get logits
            student_logits = student_outputs.logits
            teacher_logits = self.get_teacher_logits(input_ids)
            
            # Compute distillation loss
            loss, kl_loss, ce_loss = self.compute_distillation_loss(
                student_logits, teacher_logits, labels
            )
        
        return loss, kl_loss, ce_loss
    
    def train(self):
        """Main training loop"""
        # Optimize data loading
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4 if self.train_dataset.cached_data is None else 0,  # No workers needed if preprocessed
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if self.device.type == 'cuda' and self.train_dataset.cached_data is None else False,
            prefetch_factor=2 if self.train_dataset.cached_data is None else None,
            drop_last=False  # Don't drop last batch for gradient accumulation
        )
        
        global_step = 0
        total_loss = torch.tensor(0.0, device=self.device)  # Keep as tensor to avoid .item() calls
        
        logger.info("Starting training...")
        
        for epoch in range(self.num_epochs):
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            logger.info(f"{'='*80}")
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            
            # Track losses as tensors to avoid frequent .item() calls
            epoch_loss_tensor = torch.tensor(0.0, device=self.device)
            epoch_kl_loss_tensor = torch.tensor(0.0, device=self.device)
            epoch_ce_loss_tensor = torch.tensor(0.0, device=self.device)
            
            for step, batch in enumerate(progress_bar):
                # Training step (autocast is handled inside train_step)
                loss, kl_loss, ce_loss = self.train_step(batch)
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with autocast handling
                if self.scaler is not None and self.amp_dtype == torch.float16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.scaler is not None and self.amp_dtype == torch.float16:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)  # Faster zero_grad
                    global_step += 1
                    
                    # Logging (avoid .item() calls in hot path)
                    if global_step % self.logging_steps == 0:
                        # Convert tensor to float only when logging
                        avg_loss = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss / self.logging_steps
                        kl_val = kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
                        ce_val = ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss
                        logger.info(
                            f"Step {global_step}: Loss={avg_loss:.4f}, "
                            f"KL={kl_val:.4f}, CE={ce_val:.4f}, "
                            f"LR={self.scheduler.get_last_lr()[0]:.2e}"
                        )
                        total_loss = torch.tensor(0.0, device=self.device)
                    else:
                        # Accumulate loss value as tensor (no .item() call)
                        if isinstance(total_loss, torch.Tensor):
                            total_loss = total_loss + loss.detach() * self.gradient_accumulation_steps
                        else:
                            total_loss = loss.detach() * self.gradient_accumulation_steps
                
                # Accumulate epoch losses as tensors (no .item() calls)
                epoch_loss_tensor = epoch_loss_tensor + loss.detach() * self.gradient_accumulation_steps
                epoch_kl_loss_tensor = epoch_kl_loss_tensor + kl_loss.detach()
                epoch_ce_loss_tensor = epoch_ce_loss_tensor + ce_loss.detach()
                
                # Update progress bar only every few steps to reduce synchronization overhead
                # Update on optimizer steps or every 10 steps, whichever is more frequent
                update_progress = (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) % 10 == 0
                if update_progress:
                    # Only convert to Python float when updating progress bar
                    loss_val = epoch_loss_tensor.item() / (step + 1) if step > 0 else loss.item() * self.gradient_accumulation_steps
                    kl_val = epoch_kl_loss_tensor.item() / (step + 1) if step > 0 else kl_loss.item()
                    ce_val = epoch_ce_loss_tensor.item() / (step + 1) if step > 0 else ce_loss.item()
                    progress_bar.set_postfix({
                        'loss': f'{loss_val:.4f}',
                        'kl': f'{kl_val:.4f}',
                        'ce': f'{ce_val:.4f}'
                    })
                
                # Save checkpoint
                if global_step % self.save_steps == 0 and global_step > 0:
                    checkpoint_dir = os.path.join(
                        self.output_dir, f"checkpoint-step-{global_step}"
                    )
                    self.save_checkpoint(checkpoint_dir)
            
            # End of epoch - convert tensors to floats only at the end
            if isinstance(epoch_loss_tensor, torch.Tensor):
                epoch_loss = epoch_loss_tensor.item() / len(train_loader)
                epoch_kl_loss = epoch_kl_loss_tensor.item() / len(train_loader)
                epoch_ce_loss = epoch_ce_loss_tensor.item() / len(train_loader)
            else:
                epoch_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} completed. Average Loss: {epoch_loss:.4f}")
            
            # Save checkpoint after each epoch
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch + 1}")
            self.save_checkpoint(checkpoint_dir)
        
        # Save final model
        logger.info("Training completed. Saving final model...")
        self.save_checkpoint(self.output_dir)
        
        logger.info(f"Model saved to {self.output_dir}")
    
    def save_checkpoint(self, checkpoint_dir):
        """Save model checkpoint"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save student model
        self.student_model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training configuration
        config = {
            'temperature': self.temperature,
            'alpha': self.alpha,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'max_length': self.max_length
        }
        
        with open(os.path.join(checkpoint_dir, 'distillation_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")


def create_student_model_config(teacher_config, student_config):
    """Create a smaller student model config based on teacher config"""
    # Create a copy of teacher config
    student_config_dict = teacher_config.to_dict()
    
    # Update with student-specific parameters
    student_config_dict.update(student_config)
    
    # Create new config from dict
    student_model_config = type(teacher_config).from_dict(student_config_dict)
    
    return student_model_config


def main():
    parser = argparse.ArgumentParser(
        description="Train a distilled model for Chinese-Japanese translation"
    )
    
    # Model paths
    parser.add_argument(
        "--teacher_model_path",
        type=str,
        default="Hunyuan-MT-7B",
        help="Path to teacher model (Hunyuan-MT-7B)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./distilled_model_cn_ja",
        help="Output directory for distilled model"
    )
    
    # Dataset paths
    parser.add_argument(
        "--train_ja_file",
        type=str,
        default="Web-Crawled-Corpus-for-Japanese-Chinese-NMT/WCC-JC 2.0/train-ja-demo-200k.txt",
        help="Path to Japanese training file"
    )
    parser.add_argument(
        "--train_zh_file",
        type=str,
        default="Web-Crawled-Corpus-for-Japanese-Chinese-NMT/WCC-JC 2.0/train-ch-demo-200k.txt",
        help="Path to Chinese training file"
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="ja2zh",
        choices=["ja2zh", "zh2ja", "both"],
        help="Translation direction: ja2zh, zh2ja, or both"
    )
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    
    # Distillation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.5,
        help="Temperature for knowledge distillation"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Weight for distillation loss (vs hard label loss)"
    )
    
    # Student model architecture
    parser.add_argument(
        "--student_hidden_size",
        type=int,
        default=2048,
        help="Hidden size of student model"
    )
    parser.add_argument(
        "--student_num_layers",
        type=int,
        default=8,
        help="Number of layers in student model"
    )
    parser.add_argument(
        "--student_num_heads",
        type=int,
        default=8,
        help="Number of attention heads in student model"
    )
    parser.add_argument(
        "--student_intermediate_size",
        type=int,
        default=8192,
        help="Intermediate size of student model"
    )
    
    # Other parameters
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Mixed precision: {args.fp16 and device.type == 'cuda'}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.teacher_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)
    
    # Load teacher model
    logger.info(f"Loading teacher model from {args.teacher_model_path}...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_path,
        device_map="auto" if device.type == "cuda" else None,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True
    )
    teacher_model.eval()
    
    # Get teacher config
    teacher_config = teacher_model.config
    
    # Create student model config
    # Calculate attention head dimension
    head_dim = args.student_hidden_size // args.student_num_heads
    attention_head_dim = head_dim  # Usually same as head_dim
    
    student_config = {
        'hidden_size': args.student_hidden_size,
        'num_hidden_layers': args.student_num_layers,
        'num_attention_heads': args.student_num_heads,
        'intermediate_size': args.student_intermediate_size,
        'num_key_value_heads': max(1, args.student_num_heads // 4),  # GQA ratio, at least 1
        'attention_head_dim': attention_head_dim,
        'head_dim': head_dim,
    }
    
    logger.info("Creating student model...")
    logger.info(f"Student config: {student_config}")
    
    # Create student model config
    student_config_obj = create_student_model_config(teacher_config, student_config)
    
    # Create student model from config
    student_model = AutoModelForCausalLM.from_config(
        student_config_obj,
        trust_remote_code=True
    )
    
    # Initialize student model weights
    student_model.init_weights()
    
    # Copy tokenizer embeddings (if compatible sizes)
    teacher_emb = teacher_model.get_input_embeddings()
    student_emb = student_model.get_input_embeddings()
    
    if teacher_emb.weight.shape[1] == student_emb.weight.shape[1]:
        # Copy embeddings for overlapping vocab
        vocab_size = min(teacher_emb.weight.shape[0], student_emb.weight.shape[0])
        student_emb.weight.data[:vocab_size] = teacher_emb.weight.data[:vocab_size].clone()
        logger.info(f"Copied embeddings for {vocab_size} tokens from teacher")
    
    # Also copy output embeddings if tied
    if teacher_config.tie_word_embeddings and student_config_obj.tie_word_embeddings:
        logger.info("Word embeddings are tied, already copied")
    
    logger.info(f"Student model created with {sum(p.numel() for p in student_model.parameters())/1e9:.2f}B parameters")
    logger.info(f"Teacher model has {sum(p.numel() for p in teacher_model.parameters())/1e9:.2f}B parameters")
    
    # Load training dataset (preprocess=True for faster training)
    logger.info("Loading training dataset...")
    train_dataset = ParallelTranslationDataset(
        args.train_ja_file,
        args.train_zh_file,
        tokenizer,
        max_length=args.max_length,
        direction=args.direction if args.direction != "both" else "ja2zh",
        preprocess=True  # Pre-tokenize for faster training
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    
    # Create trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        temperature=args.temperature,
        alpha=args.alpha,
        device=device,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_length=args.max_length,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        fp16=args.fp16 and device.type == "cuda"
    )
    
    # Start training
    trainer.train()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()

