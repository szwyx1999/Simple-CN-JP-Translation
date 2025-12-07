import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from transformers import AutoConfig
import json
from tqdm import tqdm
import argparse
from dataclasses import dataclass
from typing import Optional
import numpy as np
from itertools import islice


class ParallelDataset(Dataset):
    """Dataset for Japanese-Chinese parallel texts"""
    def __init__(self, ja_file, zh_file, tokenizer, max_length=512):
        with open(ja_file, 'r', encoding='utf-8') as f:
            self.ja_lines = [line.strip() for line in f.readlines()]
        with open(zh_file, 'r', encoding='utf-8') as f:
            self.zh_lines = [line.strip() for line in f.readlines()]
        
        assert len(self.ja_lines) == len(self.zh_lines), \
            "Japanese and Chinese files must have the same number of lines"
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.ja_lines)
    
    def __getitem__(self, idx):
        ja_text = self.ja_lines[idx]
        zh_text = self.zh_lines[idx]
        
        # Format as translation task using chat template
        # For Japanese to Chinese: using the prompt format from the model
        messages = [
            {"role": "user", "content": f"Translate the following segment into Chinese, without additional explanation.\n\n{ja_text}"},
            {"role": "assistant", "content": zh_text}
        ]

        # Apply chat template and tokenize
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        ).squeeze(0)
        
        # # Find where the assistant response starts (after <|extra_0|> token)
        # # The extra_0 token marks the start of assistant response
        # extra_0_id = self.tokenizer.convert_tokens_to_ids("<|extra_0|>")
        # if extra_0_id is None or extra_0_id == self.tokenizer.unk_token_id:
        #     # Fallback: try to find the token ID directly
        #     try:
        #         extra_0_id = self.tokenizer.encode("<|extra_0|>", add_special_tokens=False)[0]
        #     except:
        #         # If we can't find it, use a different strategy
        #         # Tokenize separately to find the boundary
        #         user_only = self.tokenizer.apply_chat_template(
        #             [messages[0]],
        #             tokenize=True,
        #             add_generation_prompt=False,
        #             return_tensors="pt"
        #         ).squeeze(0)
        #         prompt_len = len(user_only)
        # else:
        #     # Find the position of extra_0 token
        #     prompt_len = None
        #     for i, token_id in enumerate(tokenized):
        #         if token_id == extra_0_id:
        #             prompt_len = i + 1  # Start predicting after extra_0
        #             break
        
        # Tokenize user message separately to get accurate length
        user_only = self.tokenizer.apply_chat_template(
            [messages[0]],
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        ).squeeze(0)
        prompt_len = len(user_only)
        
        # Truncate if needed
        input_len = len(tokenized)
        if input_len > self.max_length:
            tokenized = tokenized[:self.max_length]
        
        # Pad to max_length
        if len(tokenized) < self.max_length:
            padding_length = self.max_length - len(tokenized)
            padding = torch.full(
                (padding_length,),
                self.tokenizer.pad_token_id,
                dtype=tokenized.dtype
            )
            tokenized = torch.cat([tokenized, padding])
        
        attention_mask = (tokenized != self.tokenizer.pad_token_id).long()
        
        # Create labels: only predict target tokens (assistant response)
        labels = tokenized.clone()
        # Remove tokens before the assistant response
        labels[:prompt_len] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': tokenized,
            'attention_mask': attention_mask,
            'labels': labels,
            'source_text': ja_text,
            'target_text': zh_text,
            'prompt_len': torch.tensor(prompt_len, dtype=torch.long),
            'was_truncated': input_len > self.max_length
        }


@dataclass
class DistillationArguments:
    """Arguments for knowledge distillation training"""
    teacher_model_path: str = "Hunyuan-MT-7B"
    student_model_path: Optional[str] = None
    output_dir: str = "./distilled_model"
    
    # Dataset paths
    train_ja_file: str = "Web-Crawled-Corpus-for-Japanese-Chinese-NMT/train data/train-cj-demo-100000.ja.txt"
    train_zh_file: str = "Web-Crawled-Corpus-for-Japanese-Chinese-NMT/train data/train-cj-demo-100000.ch.txt"
    val_ja_file: str = "Web-Crawled-Corpus-for-Japanese-Chinese-NMT/dev data/WCC-JC/valid-ja.txt"
    val_zh_file: str = "Web-Crawled-Corpus-for-Japanese-Chinese-NMT/dev data/WCC-JC/valid-zh.txt"
    
    # Training hyperparameters
    batch_size: int = 4
    learning_rate: float = 3e-4  # Better starting LR for distillation
    num_epochs: int = 3
    max_length: int = 512
    warmup_steps: int = 500
    max_iterations_per_epoch: Optional[int] = None  # Limit iterations per epoch (None = use all data)
    
    # Distillation hyperparameters
    temperature: float = 2.5  # Reduced from 4.0 - lower temperature preserves more information
    kl_loss_weight: float = 0.5  # Weight applied to KL divergence loss
    ce_loss_weight: float = 0.5  # Weight applied to cross-entropy loss
    use_soft_labels: bool = False
    
    # Student model config (smaller than teacher)
    student_hidden_size: int = 2048
    student_num_layers: int = 12
    student_num_heads: int = 16
    student_intermediate_size: int = 8192
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True

    # Sanity-check controls
    run_sanity_check: bool = False
    sanity_check_only: bool = False
    sanity_samples: int = 2
    sanity_split: str = "train"
    sanity_max_new_tokens: int = 128
    sanity_seed: int = 42


class DistillationTrainer:
    """Trainer for knowledge distillation"""
    
    def __init__(self, args: DistillationArguments):
        self.args = args
        self.device = torch.device(args.device)
        
        if args.teacher_model_path != "None":
            print("Loading teacher model...")
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                args.teacher_model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            self.teacher_model.eval()
        else:
            print("Using only CE loss")
        
        print("Creating student model...")
        
        if args.student_model_path and os.path.exists(args.student_model_path):
            print(f"Loading student model from {args.student_model_path}")
            self.student_model = AutoModelForCausalLM.from_pretrained(
                args.student_model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            self.student_tokenizer = AutoTokenizer.from_pretrained(args.student_model_path)
        else:
            print("Initializing new student model...")
            # Create smaller student model config
            student_config = AutoConfig.from_pretrained(args.teacher_model_path)
            student_config.hidden_size = args.student_hidden_size
            student_config.num_hidden_layers = args.student_num_layers
            student_config.num_attention_heads = args.student_num_heads
            student_config.num_key_value_heads = args.student_num_heads // 4
            student_config.intermediate_size = args.student_intermediate_size
            student_config.attention_head_dim = args.student_hidden_size // args.student_num_heads
            self.student_tokenizer = self.teacher_tokenizer  # Share tokenizer

            self.student_model = AutoModelForCausalLM.from_config(student_config)
            # Initialize student embeddings from teacher (if compatible)
            self._init_student_from_teacher()

        print(f"Student model created. Parameters: {sum(p.numel() for p in self.student_model.parameters())/1e9:.2f}B")
        
        # Move student to device
        self.student_model = self.student_model.to(self.device)
        
        # Enable gradient checkpointing to save memory (slightly slower but uses less VRAM)
        if hasattr(self.student_model, 'gradient_checkpointing_enable'):
            self.student_model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing for memory savings")
        
        print("Loading datasets...")
        self.train_dataset = ParallelDataset(
            args.train_ja_file,
            args.train_zh_file,
            self.student_tokenizer,
            max_length=args.max_length
        )
        
        self.val_dataset = ParallelDataset(
            args.val_ja_file,
            args.val_zh_file,
            self.student_tokenizer,
            max_length=args.max_length
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory if args.device == "cuda" else False,
            prefetch_factor=2 if args.num_workers > 0 else None
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory if args.device == "cuda" else False,
            prefetch_factor=2 if args.num_workers > 0 else None
        )
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.01,  # Add weight decay for regularization
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Calculate total steps
        max_iters = args.max_iterations_per_epoch if args.max_iterations_per_epoch else len(self.train_loader)
        total_steps = max_iters * args.num_epochs
        
        # Use cosine annealing with warmup for better convergence
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.25  # Cosine half-cycle
        )
        
        # Track loss history for later visualization
        self.history = {
            'train_iterations': [],
            'val_epochs': []
        }
        
    def _init_student_from_teacher(self):
        """Initialize student model from teacher (for compatible layers)"""
        # Initialize embeddings from teacher (vocab size is the same)
        if hasattr(self.teacher_model, 'get_input_embeddings'):
            teacher_embeddings = self.teacher_model.get_input_embeddings()
            student_embeddings = self.student_model.get_input_embeddings()
            
            if teacher_embeddings.weight.shape == student_embeddings.weight.shape:
                student_embeddings.weight.data.copy_(teacher_embeddings.weight.data)
                print("Initialized student embeddings from teacher")
            else:
                # If dimensions differ, use a projection or copy what we can
                min_dim = min(teacher_embeddings.weight.shape[1], student_embeddings.weight.shape[1])
                student_embeddings.weight.data[:, :min_dim].copy_(
                    teacher_embeddings.weight.data[:, :min_dim]
                )
                print(f"Partially initialized student embeddings (dim {min_dim})")
        
        # Initialize output layer if tied
        if hasattr(self.teacher_model, 'get_output_embeddings') and \
           hasattr(self.student_model, 'get_output_embeddings'):
            teacher_output = self.teacher_model.get_output_embeddings()
            student_output = self.student_model.get_output_embeddings()
            
            if teacher_output is not None and student_output is not None:
                if teacher_output.weight.shape == student_output.weight.shape:
                    student_output.weight.data.copy_(teacher_output.weight.data)
                    print("Initialized student output embeddings from teacher")
        
        # Initialize layer norms from teacher (first and last layers)
        teacher_layers = self.teacher_model.model.layers if hasattr(self.teacher_model, 'model') else None
        student_layers = self.student_model.model.layers if hasattr(self.student_model, 'model') else None
        
        if teacher_layers is not None and student_layers is not None:
            # Initialize first layer from teacher's first layer
            if len(teacher_layers) > 0 and len(student_layers) > 0:
                try:
                    # Copy input layer norm
                    if hasattr(teacher_layers[0], 'input_layernorm') and \
                       hasattr(student_layers[0], 'input_layernorm'):
                        if teacher_layers[0].input_layernorm.weight.shape == \
                           student_layers[0].input_layernorm.weight.shape:
                            student_layers[0].input_layernorm.weight.data.copy_(
                                teacher_layers[0].input_layernorm.weight.data
                            )
                            student_layers[0].input_layernorm.bias.data.copy_(
                                teacher_layers[0].input_layernorm.bias.data
                            )
                    
                    # Copy post-attention layer norm
                    if hasattr(teacher_layers[0], 'post_attention_layernorm') and \
                       hasattr(student_layers[0], 'post_attention_layernorm'):
                        if teacher_layers[0].post_attention_layernorm.weight.shape == \
                           student_layers[0].post_attention_layernorm.weight.shape:
                            student_layers[0].post_attention_layernorm.weight.data.copy_(
                                teacher_layers[0].post_attention_layernorm.weight.data
                            )
                            student_layers[0].post_attention_layernorm.bias.data.copy_(
                                teacher_layers[0].post_attention_layernorm.bias.data
                            )
                    
                    print("Initialized student first layer from teacher")
                except Exception as e:
                    print(f"Could not initialize layers from teacher: {e}")
        
        # Initialize model norm if it exists
        if hasattr(self.teacher_model, 'model') and hasattr(self.teacher_model.model, 'norm') and \
           hasattr(self.student_model, 'model') and hasattr(self.student_model.model, 'norm'):
            try:
                teacher_norm = self.teacher_model.model.norm
                student_norm = self.student_model.model.norm
                if teacher_norm.weight.shape == student_norm.weight.shape:
                    student_norm.weight.data.copy_(teacher_norm.weight.data)
                    if hasattr(teacher_norm, 'bias') and hasattr(student_norm, 'bias'):
                        student_norm.bias.data.copy_(teacher_norm.bias.data)
                    print("Initialized student model norm from teacher")
            except Exception as e:
                print(f"Could not initialize model norm: {e}")
    
    def get_teacher_logits(self, input_ids, attention_mask):
        """Get teacher model logits for distillation"""
        with torch.no_grad():
            # Move input to teacher device efficiently
            teacher_input = input_ids.to(self.teacher_model.device, non_blocking=True)
            teacher_attention_mask = attention_mask.to(self.teacher_model.device, non_blocking=True)

            # Teacher is already in bfloat16, no need for autocast
            outputs = self.teacher_model(
                input_ids=teacher_input,
                attention_mask=teacher_attention_mask,
                labels=None,
                output_hidden_states=False
            )
            
            # Extract logits (full sequence)
            logits = outputs.logits
            
            return logits
    
    def compute_distillation_loss(self, student_logits, labels, temperature, teacher_logits=None):
        """Compute knowledge distillation loss with improved masking"""
        # Shift logits for next token prediction
        # Student logits: shape [batch, seq_len, vocab]
        # Teacher logits: shape [batch, seq_len, vocab]
        # Labels: shape [batch, seq_len]
        
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        if teacher_logits is not None:
            shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
            teacher_probs = F.softmax(shift_teacher_logits / temperature, dim=-1)
        shift_labels = labels[..., 1:].contiguous()
        
        # Create mask for valid tokens (ignore padding and -100)
        # Only compute loss on target tokens (not prompt tokens)
        mask = (shift_labels != -100) & (shift_labels != self.student_tokenizer.pad_token_id)
        mask = mask.float()  # [batch, seq_len-1]
        
        # Check if we have any valid tokens
        num_valid_tokens = mask.sum().clamp(min=1.0)
        
        if num_valid_tokens <= 1:
            # No valid tokens, return zero loss (shouldn't happen, but safety check)
            return torch.tensor(0.0, device=student_logits.device), \
                   torch.tensor(0.0, device=student_logits.device), \
                   torch.tensor(0.0, device=student_logits.device)
        
        # Compute KL divergence loss (soft labels) - only on valid tokens
        student_log_probs = F.log_softmax(shift_student_logits / temperature, dim=-1)
        
        # Compute KL divergence
        if teacher_logits is not None:
            kl_div = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='none',
                log_target=False
            )  # [batch, seq_len-1, vocab]
        
            kl_div = kl_div.sum(dim=-1)  # [batch, seq_len-1]
            kl_div = kl_div * mask  # Apply mask to only valid tokens
            
            # Average over valid tokens (temperature^2 scaling factor from original paper)
            kl_loss = (temperature ** 2) * kl_div.sum() / num_valid_tokens
        else:
            kl_loss = torch.tensor(0.0, device=student_logits.device)
        
        # Compute cross-entropy loss (hard labels) - only on valid tokens
        # Flatten for cross-entropy
        flat_logits = shift_student_logits.view(-1, shift_student_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        flat_mask = mask.view(-1)
        
        # Compute CE loss only on valid tokens
        ce_loss_per_token = F.cross_entropy(
            flat_logits,
            flat_labels,
            reduction='none',
            ignore_index=-100
        )
        
        # Average over valid tokens only
        ce_loss = (ce_loss_per_token * flat_mask).sum() / num_valid_tokens
        
        # Combine losses with weighted sum
        if teacher_logits is not None:
            total_loss = self.args.kl_loss_weight * kl_loss + self.args.ce_loss_weight * ce_loss
        else:
            total_loss = ce_loss
        
        return total_loss, kl_loss, ce_loss

    def _masked_ce_loss(self, logits, labels):
        """Cross-entropy on next-token prediction with label masking."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        return loss

    def _decode_label_tokens(self, label_tensor):
        """Convert masked label tensor back to readable text."""
        pad_id = self.student_tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.student_tokenizer.eos_token_id
        cleaned = label_tensor.clone()
        cleaned[cleaned == -100] = pad_id
        cleaned = cleaned[cleaned != pad_id]
        if cleaned.numel() == 0:
            return ""
        return self.student_tokenizer.decode(cleaned, skip_special_tokens=True).strip()
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.student_model.train()
        total_loss = 0
        total_kl_loss = 0
        total_ce_loss = 0
        
        # Determine max iterations
        max_iters = self.args.max_iterations_per_epoch
        if max_iters is not None:
            # Use itertools.islice to limit the iterator
            loader_iter = islice(self.train_loader, max_iters)
            num_batches = min(max_iters, len(self.train_loader))
        else:
            loader_iter = self.train_loader
            num_batches = len(self.train_loader)
        
        progress_bar = tqdm(loader_iter, desc=f"Epoch {epoch+1}", total=num_batches)
        self.optimizer.zero_grad()  # Zero gradients at the start
        
        num_batches_processed = 0
        for batch_idx, batch in enumerate(progress_bar):
            num_batches_processed += 1
            # Move to device with non_blocking for faster transfer
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # Forward pass through student
            student_outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None
            )
            student_logits = student_outputs.logits
            
            # compute loss
            if self.args.use_soft_labels and self.teacher_model is not None:
                teacher_logits = self.get_teacher_logits(input_ids, attention_mask)
                teacher_logits = teacher_logits.to(self.device, non_blocking=True)
            else:
                teacher_logits = None
                
            loss, kl_loss, ce_loss = self.compute_distillation_loss(
                student_logits,
                labels,
                self.args.temperature,
                teacher_logits
            )
            kl_value = kl_loss.item()
            ce_value = ce_loss.item()
            total_kl_loss += kl_value
            total_ce_loss += ce_value
            
            loss_value = loss.item()
            self._record_iteration(epoch, batch_idx, loss_value, kl_value, ce_value)
            loss.backward()
            
            total_loss += loss_value
            
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                avg_loss_so_far = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({
                    'kl_loss': kl_value,
                    'ce_loss': ce_value,
                    'loss': loss_value,
                    'avg_loss': avg_loss_so_far,
                })
        
        avg_loss = total_loss / num_batches_processed if num_batches_processed > 0 else 0.0
        avg_kl_loss = total_kl_loss / num_batches_processed if (self.args.use_soft_labels and num_batches_processed > 0) else 0
        avg_ce_loss = total_ce_loss / num_batches_processed if (self.args.use_soft_labels and num_batches_processed > 0) else 0
        
        return avg_loss, avg_kl_loss, avg_ce_loss
    
    def validate(self):
        """Validate the model"""
        self.student_model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Compute loss manually to ensure consistency
                outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None
                )
                student_logits = outputs.logits
                
                # Shift logits and labels
                shift_logits = student_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self):
        """Main training loop"""
        if self.args.sanity_check_only:
            print("Sanity check only flag detected. Skipping training.")
            return

        print("Starting training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.args.num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.args.num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss, kl_loss, ce_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")
            if self.args.use_soft_labels:
                print(f"  KL Loss: {kl_loss:.4f}")
                print(f"  CE Loss: {ce_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            print(f"Val Loss: {val_loss:.4f}")

            # Record per-epoch summary
            self.history['val_epochs'].append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'kl_loss': kl_loss if self.args.use_soft_labels else None,
                'ce_loss': ce_loss if self.args.use_soft_labels else None,
                'val_loss': val_loss
            })
            self.save_history()
            
            # Save checkpoint
            checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch-{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            self.student_model.save_pretrained(checkpoint_dir)
            self.student_tokenizer.save_pretrained(checkpoint_dir)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"New best model! Saving to {self.args.output_dir}")
                self.student_model.save_pretrained(self.args.output_dir)
                self.student_tokenizer.save_pretrained(self.args.output_dir)
                
                # Save training config
                config_dict = {
                    'teacher_model': self.args.teacher_model_path,
                    'student_config': {
                        'hidden_size': self.args.student_hidden_size,
                        'num_layers': self.args.student_num_layers,
                        'num_heads': self.args.student_num_heads,
                        'intermediate_size': self.args.student_intermediate_size
                    },
                    'training_args': {
                        'temperature': self.args.temperature,
                        'kl_loss_weight': self.args.kl_loss_weight,
                        'ce_loss_weight': self.args.ce_loss_weight,
                        'learning_rate': self.args.learning_rate,
                        'batch_size': self.args.batch_size,
                        'num_epochs': self.args.num_epochs
                    }
                }
                
                with open(os.path.join(self.args.output_dir, "distillation_config.json"), 'w') as f:
                    json.dump(config_dict, f, indent=2)
        
        print("\nTraining completed!")
        self.save_history()

    def _record_iteration(self, epoch, batch_idx, loss_value, kl_value, ce_value):
        """Store per-iteration loss information for later analysis."""
        self.history['train_iterations'].append({
            'epoch': epoch + 1,
            'batch_index': batch_idx + 1,
            'global_step': len(self.history['train_iterations']) + 1,
            'loss': float(loss_value),
            'kl_loss': float(kl_value) if kl_value is not None else None,
            'ce_loss': float(ce_value) if ce_value is not None else None
        })

    def save_history(self):
        """Persist loss history to disk for external visualization."""
        if not self.history['train_iterations'] and not self.history['val_epochs']:
            return

        os.makedirs(self.args.output_dir, exist_ok=True)
        history_path = os.path.join(self.args.output_dir, "loss_history.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved loss history to {history_path}")


def main():
    parser = argparse.ArgumentParser(description="Train distilled model")
    parser.add_argument("--teacher_model_path", type=str, default="Hunyuan-MT-7B",
                        help="Path to teacher model")
    parser.add_argument("--student_model_path", type=str, default=None,
                        help="Path to student model (if continuing training)")
    parser.add_argument("--output_dir", type=str, default="./distilled_model",
                        help="Output directory for distilled model")
    parser.add_argument("--train_ja_file", type=str,
                        default="Web-Crawled-Corpus-for-Japanese-Chinese-NMT/WCC-JC 2.0/train-ja-demo-200k.txt")
    parser.add_argument("--train_zh_file", type=str,
                        default="Web-Crawled-Corpus-for-Japanese-Chinese-NMT/WCC-JC 2.0/train-ch-demo-200k.txt")
    parser.add_argument("--val_ja_file", type=str,
                        default="Web-Crawled-Corpus-for-Japanese-Chinese-NMT/dev data/WCC-JC/valid-ja.txt")
    parser.add_argument("--val_zh_file", type=str,
                        default="Web-Crawled-Corpus-for-Japanese-Chinese-NMT/dev data/WCC-JC/valid-zh.txt")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="Learning rate (recommended: 3e-4 to 5e-4 for distillation)")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=2.5,
                        help="Temperature for distillation (lower preserves more info, recommended: 2.0-3.0)")
    parser.add_argument("--kl_loss_weight", type=float, default=0.2,
                        help="Weight applied to KL divergence loss component")
    parser.add_argument("--ce_loss_weight", type=float, default=0.8,
                        help="Weight applied to cross-entropy loss component")
    parser.add_argument("--student_hidden_size", type=int, default=2048)
    parser.add_argument("--student_num_layers", type=int, default=12)
    parser.add_argument("--student_num_heads", type=int, default=8)
    parser.add_argument("--student_intermediate_size", type=int, default=8192)
    parser.add_argument("--max_iterations_per_epoch", type=int, default=10000,
                        help="Maximum number of iterations per epoch (None = use all data)")
    parser.add_argument("--use_soft_labels", action="store_true",
                        help="Use soft labels for distillation")
    parser.add_argument("--run_sanity_check", action="store_true",
                        help="Run a small sanity-check pass before training")
    parser.add_argument("--sanity_check_only", action="store_true",
                        help="Only run the sanity check and skip training")
    parser.add_argument("--sanity_split", type=str, default="train", choices=["train", "val"],
                        help="Dataset split to sample during sanity check")
    parser.add_argument("--sanity_samples", type=int, default=3,
                        help="Number of samples to inspect during sanity check")
    parser.add_argument("--sanity_max_new_tokens", type=int, default=128,
                        help="Max new tokens to decode when previewing generations")
    parser.add_argument("--sanity_seed", type=int, default=10000,
                        help="Random seed for selecting sanity-check samples")
    
    args = parser.parse_args()

    print("Training with the following arguments:")
    print(args)
    
    dist_args = DistillationArguments(
        teacher_model_path=args.teacher_model_path,
        student_model_path=args.student_model_path,
        output_dir=args.output_dir,
        train_ja_file=args.train_ja_file,
        train_zh_file=args.train_zh_file,
        val_ja_file=args.val_ja_file,
        val_zh_file=args.val_zh_file,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        temperature=args.temperature,
        kl_loss_weight=args.kl_loss_weight,
        ce_loss_weight=args.ce_loss_weight,
        student_hidden_size=args.student_hidden_size,
        student_num_layers=args.student_num_layers,
        student_num_heads=args.student_num_heads,
        student_intermediate_size=args.student_intermediate_size,
        max_iterations_per_epoch=args.max_iterations_per_epoch,
        run_sanity_check=args.run_sanity_check,
        sanity_check_only=args.sanity_check_only,
        sanity_samples=args.sanity_samples,
        sanity_split=args.sanity_split,
        sanity_max_new_tokens=args.sanity_max_new_tokens,
        sanity_seed=args.sanity_seed
    )
    
    trainer = DistillationTrainer(dist_args)
    
    if dist_args.run_sanity_check or dist_args.sanity_check_only:
        trainer.run_sanity_check(
            split=dist_args.sanity_split,
            num_samples=dist_args.sanity_samples,
            max_new_tokens=dist_args.sanity_max_new_tokens,
            seed=dist_args.sanity_seed
        )
        if dist_args.sanity_check_only:
            print("Sanity check completed. Exiting because --sanity_check_only was set.")
            return
    trainer.train()


if __name__ == "__main__":
    main()