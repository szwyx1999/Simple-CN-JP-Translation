"""
Knowledge Distillation Training Script for Hunyuan-MT-7B
Specialized for Chinese-Japanese Translation using WCC-JC 2.0 dataset

This version adds:
- `--resume_from` to continue training from an existing student checkpoint.
- Qwen2.5-friendly tokenizer settings (pad_token & padding_side="left").
- Default teacher model path set to Qwen/Qwen2.5-7B-Instruct.
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
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import math
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ParallelTranslationDataset(Dataset):
    """Dataset for parallel Japanese-Chinese translation"""
    
    def __init__(
        self,
        ja_file,
        zh_file,
        tokenizer,
        max_length=512,
        direction="ja2zh",
        preprocess=False,
        cache_file=None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.direction = direction
        
        # Read parallel data
        logger.info(f"Loading parallel data from {ja_file} and {zh_file}")
        with open(ja_file, 'r', encoding='utf-8') as f:
            self.ja_lines = [line.strip() for line in f.readlines()]
        
        with open(zh_file, 'r', encoding='utf-8') as f:
            self.zh_lines = [line.strip() for line in f.readlines()]
        
        assert len(self.ja_lines) == len(self.zh_lines), "Parallel files must have the same number of lines"
        
        logger.info(f"Loaded {len(self.ja_lines)} parallel sentences")
        
        # Preprocess and cache if requested
        self.cached_data = None
        if preprocess:
            if cache_file and os.path.exists(cache_file):
                logger.info(f"Loading preprocessed data from cache: {cache_file}")
                self.cached_data = torch.load(cache_file)
            else:
                logger.info("Preprocessing and tokenizing dataset...")
                self.cached_data = []
                for idx in tqdm(range(len(self.ja_lines)), desc="Pre-tokenizing"):
                    item = self._tokenize_item(idx)
                    self.cached_data.append(item)
                
                if cache_file:
                    logger.info(f"Saving preprocessed data to cache: {cache_file}")
                    torch.save(self.cached_data, cache_file)
                
                logger.info("Preprocessing complete!")
        else:
            logger.info("Will tokenize on-the-fly during training")
    
    def __len__(self):
        return len(self.ja_lines)
    
    def _tokenize_item(self, idx):
        """Tokenize a single item (used for both preprocessing and on-the-fly)"""
        ja_text = self.ja_lines[idx]
        zh_text = self.zh_lines[idx]
        
        # Determine source and target based on direction
        if self.direction == "ja2zh":
            src_text = ja_text
            tgt_text = zh_text
            src_lang = "Japanese"
            tgt_lang = "Chinese"
        else:
            src_text = zh_text
            tgt_text = ja_text
            src_lang = "Chinese"
            tgt_lang = "Japanese"
        
        # Build prompt using chat template
        # User message: source sentence with explicit translation instruction
        # Assistant message: target sentence (used as label)
        messages = [
            {
                "role": "user",
                "content": f"Translate the following {src_lang} sentence into {tgt_lang}:\n\n{src_text}"
            },
            {
                "role": "assistant",
                "content": tgt_text
            }
        ]
        
        try:
            # Use chat template if available (Qwen2.5, etc.)
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception:
            # Fallback: simple concatenation
            prompt = f"User: Translate the following {src_lang} sentence into {tgt_lang}:\n{src_text}\nAssistant:"
        
        # Tokenize
        model_inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize target separately to determine where labels start
        with self.tokenizer.as_target_tokenizer():
            target_tokens = self.tokenizer(
                tgt_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                add_special_tokens=False
            )
        
        # Create labels: -100 for prompt tokens, target ids for answer
        input_ids = model_inputs["input_ids"][0]
        labels = torch.full_like(input_ids, -100)
        
        # Simple heuristic: assume target tokens are at the end
        target_len = len(target_tokens["input_ids"])
        labels[-target_len:] = torch.tensor(target_tokens["input_ids"][-target_len:], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": labels
        }
    
    def __getitem__(self, idx):
        if self.cached_data is not None:
            return self.cached_data[idx]
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
        gradient_accumulation_steps=4,
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
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Move student model to device (GPU)
        self.student_model.to(device)
        self.student_model.train()
        
        # Enable optimizations for better performance
        torch.backends.cudnn.benchmark = True
        
        # Create dataloaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        if self.val_dataset is not None:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        else:
            self.val_dataloader = None
        
        # Setup optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.student_model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in self.student_model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate
        )
        
        # Setup scheduler
        num_training_steps = self.num_epochs * math.ceil(len(self.train_dataloader) / self.gradient_accumulation_steps)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
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
        
        # Compute teacher probabilities with temperature scaling
        teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
        
        # Log-softmax for student with temperature
        student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
        
        # Compute KL divergence between teacher and student
        # KLDivLoss with log_target=False expects input as log-probs and target as probs
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (temp ** 2)
        
        # Compute standard cross entropy loss w.r.t. ground truth
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        # Combine losses
        loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        
        return loss, kl_loss, ce_loss
    
    def get_teacher_logits(self, input_ids):
        """Compute teacher logits without gradient and with caching"""
        with torch.no_grad():
            outputs = self.teacher_model(
                input_ids=input_ids,
                output_hidden_states=False,
                use_cache=False  # Disable cache for faster inference
            )
        
        logits = outputs.logits
        return logits

    def train_step(self, batch):
        """Perform a single training step (optimized)"""
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        labels = batch['labels'].to(self.device, non_blocking=True)
        
        # Forward pass
        self.student_model.train()
        with torch.cuda.amp.autocast(dtype=self.amp_dtype, enabled=self.use_amp):
            student_outputs = self.student_model(
                input_ids=input_ids,
                output_hidden_states=False,
                return_dict=True
            )
            student_logits = student_outputs.logits
            
            teacher_logits = self.get_teacher_logits(input_ids)
            
            loss, kl_loss, ce_loss = self.compute_distillation_loss(
                student_logits, teacher_logits, labels
            )
        
        # Normalize by gradient accumulation steps
        loss = loss / self.gradient_accumulation_steps
        
        # Backward
        if self.use_amp and self.amp_dtype == torch.float16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.gradient_accumulation_steps, kl_loss.item(), ce_loss.item()
    
    def save_checkpoint(self, epoch, step):
        """Save model and tokenizer checkpoint"""
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch}-step-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        logger.info(f"Saving checkpoint to {checkpoint_dir}")
        
        # Save student model and tokenizer
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
    
    def train(self):
        """Main training loop"""
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(1, self.num_epochs + 1):
            logger.info(f"Starting epoch {epoch}/{self.num_epochs}")
            
            epoch_loss = 0.0
            epoch_kl_loss = 0.0
            epoch_ce_loss = 0.0
            
            self.student_model.train()
            
            optimizer = self.optimizer
            scheduler = self.lr_scheduler
            
            optimizer.zero_grad(set_to_none=True)
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch}",
                dynamic_ncols=True
            )
            
            for step, batch in enumerate(progress_bar):
                loss, kl_loss, ce_loss = self.train_step(batch)
                epoch_loss += loss
                epoch_kl_loss += kl_loss
                epoch_ce_loss += ce_loss
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.use_amp and self.amp_dtype == torch.float16:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    
                    if global_step % self.logging_steps == 0:
                        avg_loss = epoch_loss / (step + 1)
                        avg_kl = epoch_kl_loss / (step + 1)
                        avg_ce = epoch_ce_loss / (step + 1)
                        current_lr = scheduler.get_last_lr()[0]
                        
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "kl": f"{avg_kl:.4f}",
                            "ce": f"{avg_ce:.4f}",
                            "lr": f"{current_lr:.2e}"
                        })
                        logger.info(
                            f"Step {global_step} | "
                            f"loss={avg_loss:.4f}, kl={avg_kl:.4f}, ce={avg_ce:.4f}, lr={current_lr:.2e}"
                        )
                    
                    if global_step % self.save_steps == 0:
                        self.save_checkpoint(epoch, global_step)
            
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            avg_epoch_kl = epoch_kl_loss / len(self.train_dataloader)
            avg_epoch_ce = epoch_ce_loss / len(self.train_dataloader)
            logger.info(
                f"Epoch {epoch} completed | "
                f"avg_loss={avg_epoch_loss:.4f}, avg_kl={avg_epoch_kl:.4f}, avg_ce={avg_epoch_ce:.4f}"
            )
            
            # Save checkpoint at end of epoch
            self.save_checkpoint(epoch, global_step)


# def create_student_model_config(teacher_config, student_config):
#     """Create a smaller student model config based on teacher config"""
#     student_config_dict = teacher_config.to_dict()
#     student_config_dict.update(student_config)
#     new_config = AutoConfig.from_dict(student_config_dict)
#     return new_config

def create_student_model_config(teacher_config, student_config):
    """Create a smaller student model config based on teacher config.

    We start from teacher_config.to_dict(), update the fields for the student,
    and then rebuild a config using the same config class as the teacher
    (e.g., Qwen2Config), instead of AutoConfig. We also reset `layer_types`
    so that Qwen2Config can regenerate a consistent pattern.
    """
    # get full teacher config as dict
    student_config_dict = teacher_config.to_dict()
    # update the fields for the student
    student_config_dict.update(student_config)

    # key step: remove layer_types so that Qwen2Config can regenerate it by num_hidden_layers
    if "layer_types" in student_config_dict:
        student_config_dict["layer_types"] = None

    # rebuild a config using the same config class as the teacher
    new_config = teacher_config.__class__.from_dict(student_config_dict)
    return new_config



def main():
    parser = argparse.ArgumentParser(
        description="Train a distilled model for Chinese-Japanese translation"
    )
    
    # Model paths
    parser.add_argument(
        "--teacher_model_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Path to teacher model (e.g., Qwen/Qwen2.5-7B-Instruct)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./distilled_model_cn_ja",
        help="Output directory for distilled model"
    )
    
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to an existing student checkpoint to resume training from"
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
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Evaluate every X steps (currently unused)"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log training progress every X steps"
    )
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
    
    # Load teacher model
    logger.info(f"Loading teacher model from {args.teacher_model_path}...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_path,
        device_map="auto" if device.type == "cuda" else None,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    teacher_model.eval()
    
    # Get teacher config
    teacher_config = teacher_model.config
    
    # Load tokenizer and student model (optionally resume from checkpoint)
    if args.resume_from:
        logger.info(f"Resuming student model from checkpoint: {args.resume_from}")
        # load student & config from checkpoint
        student_model = AutoModelForCausalLM.from_pretrained(
            args.resume_from,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
        )

        # load tokenizer only from teacher model
        logger.info(f"Loading tokenizer from teacher model: {args.teacher_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.teacher_model_path,
            trust_remote_code=True,
        )
    else:
        logger.info(f"Loading tokenizer from {args.teacher_model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.teacher_model_path,
            trust_remote_code=True,
        )
        # Ensure pad_token and left padding are set for decoder-only models like Qwen
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        # Create student model config
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
        
        # Create student model from config
        student_config_obj = create_student_model_config(teacher_config, student_config)
        student_model = AutoModelForCausalLM.from_config(
            student_config_obj,
            trust_remote_code=True,
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
        
        # If word embeddings are tied in both models, weights are already shared
        if getattr(teacher_config, 'tie_word_embeddings', False) and getattr(student_config_obj, 'tie_word_embeddings', False):
            logger.info("Word embeddings are tied between input and output.")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    logger.info(f"Student model has {sum(p.numel() for p in student_model.parameters())/1e9:.2f}B parameters")
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
