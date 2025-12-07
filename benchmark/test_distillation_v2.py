import os
import json
import argparse
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF, TER


class ParallelDataset(Dataset):
    """
    Simple Japanese–Chinese parallel dataset for evaluation.

    Each item returns:
        - input_ids: tokenized prompt for the source sentence
        - source_text: raw Japanese sentence
        - target_text: raw Chinese reference sentence
    """

    def __init__(self, ja_file: str, zh_file: str, tokenizer, max_length: int = 512) -> None:
        with open(ja_file, "r", encoding="utf-8") as f_ja:
            self.ja_lines = [line.strip() for line in f_ja]

        with open(zh_file, "r", encoding="utf-8") as f_zh:
            self.zh_lines = [line.strip() for line in f_zh]

        if len(self.ja_lines) != len(self.zh_lines):
            raise ValueError(
                f"Mismatched line counts: {len(self.ja_lines)} Japanese vs {len(self.zh_lines)} Chinese"
            )

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.ja_lines)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ja_text = self.ja_lines[idx]
        zh_text = self.zh_lines[idx]

        # Prompt: align with training script style
        # Training used:
        #   "Translate the following {src_lang} sentence into {tgt_lang}:\n\n{src_text}"
        # Here src_lang = Japanese, tgt_lang = Chinese.
        prompt = f"Translate the following Japanese sentence into Chinese:\n\n{ja_text}"
        messages = [{"role": "user", "content": prompt}]

        # Use chat template with generation prompt so the model generates the assistant turn
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).squeeze(0)

        # Truncate if needed (keep right-most tokens, important for left padding)
        if tokenized.size(0) > self.max_length:
            tokenized = tokenized[-self.max_length:]

        # Pad to max_length (left PAD – matches training)
        if tokenized.size(0) < self.max_length:
            pad_len = self.max_length - tokenized.size(0)
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            padding = torch.full((pad_len,), pad_id, dtype=torch.long)
            tokenized = torch.cat([padding, tokenized], dim=0)

        return {
            "input_ids": tokenized,
            "source_text": ja_text,
            "target_text": zh_text,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to stack input_ids and keep texts as lists.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    source_text = [item["source_text"] for item in batch]
    target_text = [item["target_text"] for item in batch]
    return {
        "input_ids": input_ids,
        "source_text": source_text,
        "target_text": target_text,
    }


def extract_translation(gen_text: str, source_text: str) -> str:
    """
    Try to extract only the translation part from the raw generated text.

    Heuristics (ordered):
      1. If a chat-style assistant block is present ("<|im_start|>assistant"),
         take the content after it until the next end marker.
      2. Else, if special marker "<|extra_0|>" exists, take text after it.
      3. Cut at common EOS / end markers if present.
      4. If the source_text appears in the output, take text after it.
      5. Strip some leftover special tokens.
    """
    text = gen_text

    # 1) Qwen-style chat: grab content after assistant tag
    if "<|im_start|>assistant" in text:
        parts = text.split("<|im_start|>assistant", maxsplit=1)
        text = parts[1] if len(parts) > 1 else parts[0]

    # 2) Older marker used in some prompts
    if "<|extra_0|>" in text:
        parts = text.split("<|extra_0|>", maxsplit=1)
        text = parts[1] if len(parts) > 1 else parts[0]

    # 3) Cut at several possible EOS-like markers
    for eos_tok in ["<|im_end|>", "<|eos|>", "<|endoftext|>"]:
        if eos_tok in text:
            text = text.split(eos_tok)[0]

    # 4) If model echoed the source sentence, drop everything up to it
    if source_text and source_text in text:
        text = text.split(source_text)[-1]

    # 5) Remove some common leftover special tokens
    for token in ["<|startoftext|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]:
        text = text.replace(token, "")

    return text.strip()


# Set up SacreBLEU metrics with proper Chinese handling
_bleu_metric = BLEU(tokenize="zh")          # Chinese-aware BLEU tokenization
_chrf_metric = CHRF()                       # chrF is already character-based
_ter_metric = TER(asian_support=True)       # Treat CJK characters appropriately


def compute_bleu(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute corpus BLEU (Chinese-aware tokenization).
    """
    score = _bleu_metric.corpus_score(hypotheses, [references])
    return float(score.score)


def compute_chrf(references: List[str], hypotheses: List[str]) -> float:
    score = _chrf_metric.corpus_score(hypotheses, [references])
    return float(score.score)


def compute_ter(references: List[str], hypotheses: List[str]) -> float:
    score = _ter_metric.corpus_score(hypotheses, [references])
    return float(score.score)


def evaluate_model(
    model,
    tokenizer,
    test_dataset: ParallelDataset,
    device: torch.device,
    batch_size: int = 8,
    max_new_tokens: int = 256,
) -> Dict[str, List[str]]:
    """
    Run deterministic generation on the test set and collect references/hypotheses.
    """
    model.eval()
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    all_references: List[str] = []
    all_hypotheses: List[str] = []

    print("Evaluating model...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Batches"):
            input_ids = batch["input_ids"].to(device)
            references = batch["target_text"]
            sources = batch["source_text"]

            # Deterministic generation (no sampling)
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy decoding
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            generated_texts = tokenizer.batch_decode(
                outputs, skip_special_tokens=False
            )

            for src, ref, gen_text in zip(sources, references, generated_texts):
                hyp = extract_translation(gen_text, src)
                all_references.append(ref)
                all_hypotheses.append(hyp)

    return {"references": all_references, "hypotheses": all_hypotheses}


def save_examples(
    references: List[str],
    hypotheses: List[str],
    output_file: str,
    num_examples: int = 20,
) -> None:
    """Save a subset of examples for manual inspection."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Examples of translations:\n")
        f.write("=" * 80 + "\n\n")

        for i in range(min(num_examples, len(references))):
            f.write(f"Example {i + 1}:\n")
            f.write(f"Reference: {references[i]}\n")
            f.write(f"Hypothesis: {hypotheses[i]}\n")
            f.write("-" * 80 + "\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic evaluation of a JP->ZH translation model on WCC-JC."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./distilled_model",
        help="Path to the HF model (teacher or student).",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help=(
            "Path or name of the tokenizer to use. "
            "By default, uses the original teacher tokenizer "
            "(Qwen/Qwen2.5-7B-Instruct)."
        ),
    )
    parser.add_argument(
        "--test_ja_file",
        type=str,
        required=True,
        help="Path to Japanese test file (one sentence per line).",
    )
    parser.add_argument(
        "--test_zh_file",
        type=str,
        required=True,
        help="Path to Chinese reference test file (one sentence per line).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum input sequence length for prompts.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g., 'cuda', 'cuda:0', 'cpu'). "
             "If not set, will use CUDA if available.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save results (translations and scores).",
    )

    args = parser.parse_args()

    # Resolve device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Ensure output directory exists (default ./results)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer (from teacher / base model, not from distilled checkpoint)
    tokenizer_source = args.tokenizer_path
    print(f"Loading tokenizer from {tokenizer_source}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

    # For decoder-only models, use left padding and make sure pad_token is set
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto" if device.type == "cuda" else None,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    if device.type == "cpu":
        model = model.to(device)

    # Load dataset
    print("Loading test dataset...")
    test_dataset = ParallelDataset(
        args.test_ja_file,
        args.test_zh_file,
        tokenizer,
        max_length=args.max_length,
    )
    print(f"Test set size: {len(test_dataset)}")

    # Evaluate
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        device=device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    bleu = compute_bleu(results["references"], results["hypotheses"])
    chrf = compute_chrf(results["references"], results["hypotheses"])
    ter = compute_ter(results["references"], results["hypotheses"])

    print("\n=== Evaluation Results ===")
    print(f"BLEU: {bleu:.2f}")
    print(f"chrF: {chrf:.2f}")
    print(f"TER:  {ter:.2f}")

    # Save detailed outputs
    output_path = os.path.join(args.output_dir, "translations.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for ref, hyp in zip(results["references"], results["hypotheses"]):
            f.write(json.dumps({"reference": ref, "hypothesis": hyp}, ensure_ascii=False) + "\n")

    # Save scores
    scores_path = os.path.join(args.output_dir, "scores.json")
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump({"BLEU": bleu, "chrF": chrf, "TER": ter}, f, ensure_ascii=False, indent=2)

    # Save some examples
    examples_path = os.path.join(args.output_dir, "examples.txt")
    save_examples(
        results["references"],
        results["hypotheses"],
        examples_path,
        num_examples=50,
    )

    print(f"\nResults saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()

