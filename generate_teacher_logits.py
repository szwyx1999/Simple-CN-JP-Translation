import argparse
import json
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_distillation import ParallelDataset


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and cache teacher logits for knowledge distillation"
    )
    parser.add_argument(
        "--teacher_model_path",
        type=str,
        default="Hunyuan-MT-7B",
        help="Path or identifier of the teacher model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./teacher_logits",
        help="Directory to store generated logits shards and metadata",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Name of the dataset split (used in output filenames)",
    )
    parser.add_argument(
        "--ja_file",
        type=str,
        required=True,
        help="Path to the Japanese source file",
    )
    parser.add_argument(
        "--zh_file",
        type=str,
        required=True,
        help="Path to the Chinese target file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for teacher inference",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--examples_per_shard",
        type=int,
        default=256,
        help="Number of samples per saved shard file",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples to process",
    )
    parser.add_argument(
        "--inference_dtype",
        type=str,
        choices=list(DTYPE_MAP.keys()),
        default="bfloat16",
        help="Computation dtype for the teacher model forward pass (when supported)",
    )
    parser.add_argument(
        "--logits_dtype",
        type=str,
        choices=list(DTYPE_MAP.keys()),
        default="float16",
        help="Storage dtype for cached teacher logits",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (e.g., 'cuda', 'cpu'). "
             "Defaults to CUDA if available, otherwise CPU.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Pin memory in dataloader (recommended when using CUDA)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing metadata/shards if present",
    )
    return parser.parse_args()


def resolve_device(explicit_device: Optional[str] = None) -> torch.device:
    if explicit_device:
        return torch.device(explicit_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_output_dir(path: str, overwrite: bool):
    os.makedirs(path, exist_ok=True)
    metadata_candidates = [
        file for file in os.listdir(path) if file.endswith("_metadata.json")
    ]

    if metadata_candidates and not overwrite:
        raise FileExistsError(
            f"Found existing metadata files in {path}. "
            "Use --overwrite to replace them."
        )

    if overwrite:
        for filename in os.listdir(path):
            if filename.endswith(".pt") or filename.endswith("_metadata.json"):
                os.remove(os.path.join(path, filename))


def init_teacher_model(model_path: str, device: torch.device, dtype_key: str):
    dtype = DTYPE_MAP[dtype_key]
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model_kwargs = {"torch_dtype": dtype}
    if device.type == "cuda":
        model_kwargs["device_map"] = "auto"

    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )

    if device.type != "cuda":
        teacher_model = teacher_model.to(device)

    teacher_model.eval()
    return teacher_model, tokenizer


def save_shard(
    output_dir: str,
    split: str,
    shard_idx: int,
    inputs,
    labels,
    logits,
    metadata_files,
):
    if not inputs:
        return

    shard_data = {
        "input_ids": torch.stack(inputs, dim=0),
        "labels": torch.stack(labels, dim=0),
        "teacher_logits": torch.stack(logits, dim=0),
    }

    filename = f"{split}_logits_{shard_idx:05d}.pt"
    shard_path = os.path.join(output_dir, filename)
    torch.save(shard_data, shard_path)
    metadata_files.append({"filename": filename, "num_samples": len(inputs)})

    inputs.clear()
    labels.clear()
    logits.clear()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    inference_dtype = DTYPE_MAP[args.inference_dtype]
    logits_dtype = DTYPE_MAP[args.logits_dtype]

    ensure_output_dir(args.output_dir, args.overwrite)

    teacher_model, tokenizer = init_teacher_model(
        args.teacher_model_path,
        device,
        args.inference_dtype,
    )

    dataset = ParallelDataset(
        ja_file=args.ja_file,
        zh_file=args.zh_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory if device.type == "cuda" else False,
    )

    current_inputs = []
    current_labels = []
    current_logits = []

    metadata_files = []
    shard_idx = 0
    total_samples = 0
    max_samples = args.max_samples

    progress = tqdm(dataloader, desc="Generating teacher logits")

    with torch.no_grad():
        for batch in progress:
            batch_input_ids = batch["input_ids"].to(device)
            outputs = teacher_model(
                input_ids=batch_input_ids,
                output_hidden_states=False,
                labels=None,
            )

            batch_logits = outputs.logits.to(logits_dtype).cpu()
            batch_input_ids_cpu = batch["input_ids"].cpu()
            batch_labels_cpu = batch["labels"].cpu()

            batch_size = batch_input_ids_cpu.size(0)
            effective_batch = batch_size

            if max_samples is not None:
                remaining = max_samples - total_samples
                if remaining <= 0:
                    break
                effective_batch = min(batch_size, remaining)

            for idx in range(effective_batch):
                current_inputs.append(batch_input_ids_cpu[idx])
                current_labels.append(batch_labels_cpu[idx])
                current_logits.append(batch_logits[idx])
                total_samples += 1

                if len(current_inputs) >= args.examples_per_shard:
                    save_shard(
                        args.output_dir,
                        args.split,
                        shard_idx,
                        current_inputs,
                        current_labels,
                        current_logits,
                        metadata_files,
                    )
                    shard_idx += 1

            progress.set_postfix({"samples": total_samples})

            if max_samples is not None and total_samples >= max_samples:
                break

    if current_inputs:
        save_shard(
            args.output_dir,
            args.split,
            shard_idx,
            current_inputs,
            current_labels,
            current_logits,
            metadata_files,
        )

    metadata = {
        "teacher_model": args.teacher_model_path,
        "tokenizer": args.teacher_model_path,
        "split": args.split,
        "max_length": args.max_length,
        "num_samples": total_samples,
        "inference_dtype": args.inference_dtype,
        "logits_dtype": args.logits_dtype,
        "files": metadata_files,
    }

    metadata_path = os.path.join(args.output_dir, f"{args.split}_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as meta_out:
        json.dump(metadata, meta_out, indent=2)

    print(
        f"Completed generating logits for {total_samples} samples. "
        f"Metadata saved to {metadata_path}."
    )


if __name__ == "__main__":
    main()


