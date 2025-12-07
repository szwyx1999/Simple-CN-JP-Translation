from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
from typing import List


def build_messages(
    input_text: str, source_language: str, target_language: str
) -> List[dict]:
    prompt = (
        f"Translate the following {source_language} sentence into {target_language}:\n\n"
        f"{input_text}"
    )
    return [{"role": "user", "content": prompt}]


def tokenize_batch(
    texts: List[str],
    tokenizer: AutoTokenizer,
    source_language: str,
    target_language: str,
    max_length: int,
    pad_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    if not texts:
        raise ValueError("No texts provided for tokenization.")

    target_length = max_length if max_length and max_length > 0 else None
    token_tensors: List[torch.Tensor] = []

    for text in texts:
        messages = build_messages(text, source_language, target_language)
        tokenized = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            .squeeze(0)
            .to(torch.long)
        )

        if target_length is not None and tokenized.size(0) > target_length:
            tokenized = tokenized[-target_length:]

        token_tensors.append(tokenized)

    if target_length is None:
        target_length = max(t.size(0) for t in token_tensors)

    padded_tensors: List[torch.Tensor] = []
    for tokenized in token_tensors:
        if tokenized.size(0) < target_length:
            pad_len = target_length - tokenized.size(0)
            padding = torch.full((pad_len,), pad_token_id, dtype=torch.long)
            tokenized = torch.cat([padding, tokenized], dim=0)
        padded_tensors.append(tokenized)

    batch = torch.stack(padded_tensors, dim=0).to(device)
    return batch


def extract_translation(gen_text: str, source_text: str) -> str:
    """
    Mirror the deterministic extraction logic from test_distillation_v2.py
    to recover only the assistant's translation.
    """
    text = gen_text

    if "<|im_start|>assistant" in text:
        parts = text.split("<|im_start|>assistant", maxsplit=1)
        text = parts[1] if len(parts) > 1 else parts[0]

    # if "<|extra_0|>" in text:
    #     parts = text.split("<|extra_0|>", maxsplit=1)
    #     text = parts[1] if len(parts) > 1 else parts[0]

    for eos_tok in ["<|im_end|>", "<|eos|>", "<|endoftext|>"]:
        if eos_tok in text:
            text = text.split(eos_tok)[0]

    # if source_text and source_text in text:
    #     text = text.split(source_text)[-1]

    # for token in ["<|startoftext|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]:
    #     text = text.replace(token, "")

    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Inference with distilled model")
    parser.add_argument("--model_path", type=str, default="./distilled_model",
                        help="Path to distilled model")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Optional tokenizer path. Defaults to model_path if not set.")
    parser.add_argument("--input_text", type=str, default=None,
                        help="Input text to translate (optional)")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Input file with texts to translate (one per line)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file to save translations")
    parser.add_argument("--language", type=str, default="chinese",
                        help="Target language for translation")
    parser.add_argument("--source_language", type=str, default="Japanese",
                        help="Source language to use inside the prompt template")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum input length (tokens) after applying the chat template")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for batch inference")
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Loading model from {args.model_path}")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto" if device.type == "cuda" else None,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    
    if device.type == "cpu":
        model = model.to(device)
    
    model.eval()
    print("Model loaded successfully")
    
    # Generation parameters (matching teacher model settings)
    generation_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    
    target_language = args.language.strip()
    source_language = args.source_language.strip()
    
    # Handle different input modes
    if args.input_file:
        print(f"Reading input from {args.input_file}")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_texts = [line.strip() for line in f.readlines() if line.strip()]

        translations: List[str] = []
        total_batches = (len(input_texts) + args.batch_size - 1) // args.batch_size

        for batch_idx, start in enumerate(range(0, len(input_texts), args.batch_size), start=1):
            batch_texts = input_texts[start:start + args.batch_size]
            batch_input = tokenize_batch(
                batch_texts,
                tokenizer=tokenizer,
                source_language=source_language,
                target_language=target_language,
                max_length=args.max_length,
                pad_token_id=pad_token_id,
                device=device,
            )

            print(f"Translating batch {batch_idx}/{total_batches}...")
            with torch.no_grad():
                outputs = model.generate(batch_input, **generation_kwargs)

            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)

            for idx_in_batch, (source_text, decoded_text) in enumerate(zip(batch_texts, decoded_outputs), start=1):
                translation = extract_translation(decoded_text, source_text)
                translations.append(translation)
                global_idx = start + idx_in_batch
                print(f"\nInput [{global_idx}]: {source_text}")
                print(f"Translation [{global_idx}]: {translation}")

        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for translation in translations:
                    f.write(translation + "\n")
            print(f"\nTranslations saved to {args.output_file}")

    elif args.input_text:
        input_text = args.input_text.strip()
        if not input_text:
            raise ValueError("Provided input_text is empty.")

        print("Tokenizing input...")
        batch_input = tokenize_batch(
            [input_text],
            tokenizer=tokenizer,
            source_language=source_language,
            target_language=target_language,
            max_length=args.max_length,
            pad_token_id=pad_token_id,
            device=device,
        )

        print("Generating output...")
        with torch.no_grad():
            outputs = model.generate(batch_input, **generation_kwargs)

        decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        translation = extract_translation(decoded_text, input_text)

        print("\n" + "=" * 80)
        print("Input:")
        print(input_text)
        print("\nTranslation:")
        print(translation)
        print("=" * 80)

        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(translation + "\n")
            print(f"\nTranslation saved to {args.output_file}")

    else:
        print(f"\nInteractive translation mode. Target language: {target_language}")
        print("Enter text to translate (type 'quit' or 'exit' to exit):\n")

        while True:
            input_text = input("Input: ").strip()

            if input_text.lower() in ['quit', 'exit', 'q']:
                break

            if not input_text:
                continue

            batch_input = tokenize_batch(
                [input_text],
                tokenizer=tokenizer,
                source_language=source_language,
                target_language=target_language,
                max_length=args.max_length,
                pad_token_id=pad_token_id,
                device=device,
            )

            print("Translating...")
            with torch.no_grad():
                outputs = model.generate(batch_input, **generation_kwargs)

            decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            translation = extract_translation(decoded_text, input_text)
            print(f"Translation: {translation}\n")


if __name__ == "__main__":
    main()
