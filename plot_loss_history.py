import argparse
import json
import os

import numpy as np

def load_history(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"History file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_history(history, output_path: str, show: bool = False):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. "
            "Install it with `pip install matplotlib`."
        ) from exc

    train_iterations = history.get("train_iterations", [])
    if not train_iterations:
        raise ValueError("No training iteration history found in the file.")

    steps = [entry["global_step"] for entry in train_iterations]
    total_losses = [entry["loss"] for entry in train_iterations]

    # soften the loss curve
    softened_losses = []
    for i in range(0, len(total_losses)):
        softened_losses.append(np.mean(total_losses[i:i+100]))
    
    kl_losses = [
        entry["kl_loss"] for entry in train_iterations if entry["kl_loss"] is not None
    ]
    kl_steps = [
        entry["global_step"] for entry in train_iterations if entry["kl_loss"] is not None
    ]
    ce_losses = [
        entry["ce_loss"] for entry in train_iterations if entry["ce_loss"] is not None
    ]
    ce_steps = [
        entry["global_step"] for entry in train_iterations if entry["ce_loss"] is not None
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, softened_losses, label="Average Loss per Step")

    # if kl_losses:
    #     plt.plot(kl_steps, kl_losses, label="KL Loss")
    # if ce_losses:
    #     plt.plot(ce_steps, ce_losses, label="CE Loss")

    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.title("Training Loss per Iteration")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot loss history from JSON.")
    parser.add_argument(
        "--history_file",
        type=str,
        default=os.path.join("distilled_model", "loss_history.json"),
        help="Path to the JSON history file generated during training.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save the resulting plot (PNG). Defaults next to history file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving the image.",
    )
    args = parser.parse_args()

    history = load_history(args.history_file)
    output_path = (
        args.output_path
        if args.output_path
        else os.path.join(os.path.dirname(args.history_file) or ".", "loss_history.png")
    )
    plot_history(history, output_path, show=args.show)


if __name__ == "__main__":
    main()

