import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from tqdm import tqdm
import json
import sacrebleu
from collections import defaultdict


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
        
        # Format as translation task using chat template (without target for inference)
        messages = [
            {"role": "user", "content": f"Translate the following segment into Chinese, use a casual tone, without additional explanation.\n\n{ja_text}"}
        ]
        
        # Apply chat template and tokenize
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        ).squeeze(0)
        
        # Truncate if needed
        if len(tokenized) > self.max_length:
            tokenized = tokenized[:self.max_length]
        
        # Pad to max_length
        if len(tokenized) < self.max_length:
            padding_length = self.max_length - len(tokenized)
            tokenized = torch.cat([tokenized, torch.full((padding_length,), self.tokenizer.pad_token_id)])
        
        return {
            'input_ids': tokenized,
            'source_text': ja_text,
            'target_text': zh_text
        }


def compute_bleu(references, hypotheses):
    """Compute BLEU score using sacrebleu"""
    # Prepare references and hypotheses for sacrebleu
    refs = [[ref] for ref in references]
    bleu = sacrebleu.corpus_bleu(hypotheses, refs)
    return bleu.score


def evaluate_model(model, tokenizer, test_dataset, device, batch_size=8, max_new_tokens=2048):
    """Evaluate model on test dataset"""
    model.eval()
    
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_references = []
    all_hypotheses = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            references = batch['target_text']
            
            # Generate translations
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.6,
                top_k=20,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode generated texts
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Extract only the translated part (after the prompt)
            for i, gen_text in enumerate(generated_texts):
                source_text = batch['source_text'][i]
                
                # Try to extract translation after <|extra_0|> marker (assistant response start)
                if "<|extra_0|>" in gen_text:
                    parts = gen_text.split("<|extra_0|>")
                    if len(parts) > 1:
                        translation = parts[-1].split("<|eos|>")[0].strip()
                    else:
                        translation = gen_text.split("<|eos|>")[0].strip()
                elif source_text in gen_text:
                    # Fallback: extract after source text
                    translation = gen_text.split(source_text)[-1].strip()
                    translation = translation.split("<|eos|>")[0].strip()
                else:
                    # Last resort: just remove special tokens
                    translation = gen_text.split("<|eos|>")[0].strip()
                    translation = translation.replace("<|startoftext|>", "").strip()
                
                all_hypotheses.append(translation)
                all_references.append(references[i])
    
    # Compute BLEU score
    bleu_score = compute_bleu(all_references, all_hypotheses)
    
    return {
        'bleu_score': bleu_score,
        'references': all_references,
        'hypotheses': all_hypotheses
    }


def compute_chrF(references, hypotheses):
    """Compute chrF score"""
    refs = [[ref] for ref in references]
    chrf = sacrebleu.corpus_chrf(hypotheses, refs)
    return chrf.score


def compute_ter(references, hypotheses):
    """Compute TER (Translation Error Rate)"""
    refs = [[ref] for ref in references]
    ter = sacrebleu.corpus_ter(hypotheses, refs)
    return ter.score


def save_examples(references, hypotheses, output_file, num_examples=20):
    """Save example translations for manual inspection"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Examples of translations:\n")
        f.write("=" * 80 + "\n\n")
        
        for i in range(min(num_examples, len(references))):
            f.write(f"Example {i+1}:\n")
            f.write(f"Reference: {references[i]}\n")
            f.write(f"Hypothesis: {hypotheses[i]}\n")
            f.write("-" * 80 + "\n\n")


def main():
    parser = argparse.ArgumentParser(description="Test distilled model")
    parser.add_argument("--model_path", type=str, default="./distilled_model",
                        help="Path to distilled model")
    parser.add_argument("--test_ja_file", type=str,
                        default="Web-Crawled-Corpus-for-Japanese-Chinese-NMT/test data/WCC-JC/test-ja.txt",
                        help="Path to Japanese test file")
    parser.add_argument("--test_zh_file", type=str,
                        default="Web-Crawled-Corpus-for-Japanese-Chinese-NMT/test data/WCC-JC/test-zh.txt",
                        help="Path to Chinese test file")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--output_dir", type=str, default="./test_results",
                        help="Output directory for test results")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto" if device.type == "cuda" else None,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32
    )
    
    if device.type == "cpu":
        model = model.to(device)
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = ParallelDataset(
        args.test_ja_file,
        args.test_zh_file,
        tokenizer,
        max_length=args.max_length
    )
    
    print(f"Test set size: {len(test_dataset)}")
    
    # Evaluate model
    results = evaluate_model(
        model,
        tokenizer,
        test_dataset,
        device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens
    )
    
    # Compute additional metrics
    print("\nComputing additional metrics...")
    chrf_score = compute_chrF(results['references'], results['hypotheses'])
    ter_score = compute_ter(results['references'], results['hypotheses'])
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"BLEU Score: {results['bleu_score']:.2f}")
    print(f"chrF Score: {chrf_score:.2f}")
    print(f"TER Score: {ter_score:.2f}")
    print("=" * 80)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save metrics
    metrics = {
        'bleu_score': results['bleu_score'],
        'chrf_score': chrf_score,
        'ter_score': ter_score,
        'num_samples': len(results['references'])
    }
    
    with open(os.path.join(args.output_dir, "metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # Save detailed results
    detailed_results = []
    for ref, hyp in zip(results['references'], results['hypotheses']):
        detailed_results.append({
            'reference': ref,
            'hypothesis': hyp
        })
    
    with open(os.path.join(args.output_dir, "detailed_results.json"), 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    # Save example translations
    save_examples(
        results['references'],
        results['hypotheses'],
        os.path.join(args.output_dir, "examples.txt"),
        num_examples=50
    )
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
