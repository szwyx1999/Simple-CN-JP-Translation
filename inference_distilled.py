from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Inference with distilled model")
    parser.add_argument("--model_path", type=str, default="./distilled_model",
                        help="Path to distilled model")
    parser.add_argument("--input_text", type=str, default=None,
                        help="Input text to translate (optional)")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Input file with texts to translate (one per line)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file to save translations")
    parser.add_argument("--language", type=str, default="chinese",
                        help="Target language for translation")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto" if device.type == "cuda" else None,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32
    )
    
    if device.type == "cpu":
        model = model.to(device)
    
    model.eval()
    print("Model loaded successfully")
    
    # Generation parameters (matching teacher model settings)
    generation_config = {
        "do_sample": True,
        "top_k": 20,
        "top_p": 0.6,
        "repetition_penalty": 1.05,
        "temperature": 0.7,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    
    language = args.language.lower()
    
    # Handle different input modes
    if args.input_file:
        # Batch inference from file
        print(f"Reading input from {args.input_file}")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_texts = [line.strip() for line in f.readlines() if line.strip()]
        
        translations = []
        
        # Process in batches
        for i in range(0, len(input_texts), args.batch_size):
            batch_texts = input_texts[i:i + args.batch_size]
            
            # Prepare prompts
            messages_list = []
            for text in batch_texts:
                messages = [
                    {"role": "user", "content": f"Translate the following segment into {language}, use a casual tone, without additional explanation.\n\n{text}"}
                ]
                messages_list.append(messages)
            
            # Tokenize batch
            tokenized_chats = []
            for messages in messages_list:
                tokenized = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt"
                )
                tokenized_chats.append(tokenized)
            
            # Pad to same length
            max_len = max(t.size(1) for t in tokenized_chats)
            padded_chats = []
            for tokenized in tokenized_chats:
                padding = torch.zeros((1, max_len - tokenized.size(1)), dtype=torch.long)
                padded = torch.cat([tokenized, padding], dim=1)
                padded_chats.append(padded)
            
            batch_input = torch.cat(padded_chats, dim=0).to(device)
            
            # Generate
            print(f"Translating batch {i//args.batch_size + 1}/{(len(input_texts) + args.batch_size - 1)//args.batch_size}...")
            with torch.no_grad():
                outputs = model.generate(batch_input, **generation_config)
            
            # Decode and extract translations
            for j, output in enumerate(outputs):
                output_text = tokenizer.decode(output, skip_special_tokens=True)
                
                # Extract translation part
                source_text = batch_texts[j]
                prompt = f"Translate the following segment into {language}, use a casual tone, without additional explanation.\n\n{source_text}"
                
                if prompt in output_text:
                    translation = output_text.split(prompt)[-1].strip()
                    translation = translation.split("<|eos|>")[0].split("<|extra_0|>")[-1].strip()
                else:
                    translation = output_text
                
                translations.append(translation)
                print(f"\nInput [{i+j+1}]: {source_text}")
                print(f"Translation [{i+j+1}]: {translation}")
        
        # Save to output file if specified
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for translation in translations:
                    f.write(translation + "\n")
            print(f"\nTranslations saved to {args.output_file}")
    
    elif args.input_text:
        # Single text inference
        input_text = args.input_text
        
        messages = [
            {"role": "user", "content": f"Translate the following segment into {language}, use a casual tone, without additional explanation.\n\n{input_text}"}
        ]
        
        print("Tokenizing input...")
        tokenized_chat = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        )
        
        print("Generating output...")
        with torch.no_grad():
            outputs = model.generate(tokenized_chat.to(device), **generation_config)
        
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the translated part
        if input_text in output_text:
            translation = output_text.split(input_text)[-1].strip()
            translation = translation.split("<|eos|>")[0].split("<|extra_0|>")[-1].strip()
        else:
            translation = output_text
        
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
        # Interactive mode
        print(f"\nInteractive translation mode. Target language: {language}")
        print("Enter text to translate (type 'quit' or 'exit' to exit):\n")
        
        while True:
            input_text = input("Input: ").strip()
            
            if input_text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not input_text:
                continue
            
            messages = [
                {"role": "user", "content": f"Translate the following segment into {language}, use a casual tone, without additional explanation.\n\n{input_text}"}
            ]
            
            tokenized_chat = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt"
            )
            
            print("Translating...")
            with torch.no_grad():
                outputs = model.generate(tokenized_chat.to(device), **generation_config)
            
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract translation
            if input_text in output_text:
                translation = output_text.split(input_text)[-1].strip()
                translation = translation.split("<|eos|>")[0].split("<|extra_0|>")[-1].strip()
            else:
                # Try to find the translation after the prompt
                parts = output_text.split("<|extra_0|>")
                if len(parts) > 1:
                    translation = parts[-1].split("<|eos|>")[0].strip()
                else:
                    translation = output_text
            
            print(f"Translation: {translation}\n")


if __name__ == "__main__":
    main()
