"""
main.py
=======
Unified entry point for the Meme & Sarcasm Understanding project.

Commands:
    setup      → Generate sample dataset + preprocess
    train      → Train one or all models
    evaluate   → Evaluate trained checkpoints
    demo       → Run a quick end-to-end demo on a single sample

Examples:
    python main.py setup
    python main.py train --model all --epochs 10
    python main.py evaluate --model all
    python main.py demo --image path/to/meme.jpg --text "Oh great, another Monday"

Author: Lab Project – Meme & Sarcasm Understanding
"""

import argparse
import sys
import subprocess
from pathlib import Path


def cmd_setup(args):
    """Generate sample dataset and run preprocessing."""
    print("\n[SETUP] Generating sample dataset (200 samples) …")
    subprocess.run(
        [sys.executable, "src/preprocessing.py",
         "--generate_sample", f"--n_samples={args.n_samples}"],
        check=True,
    )
    print("[SETUP] Done. Dataset available in data/sample_dataset/")


def cmd_train(args):
    """Launch training pipeline."""
    cmd = [
        sys.executable, "src/train.py",
        f"--model={args.model}",
        f"--epochs={args.epochs}",
        f"--batch_size={args.batch_size}",
        f"--lr={args.lr}",
        f"--patience={args.patience}",
        f"--seed={args.seed}",
    ]
    if args.use_mmsd2:
        cmd.append("--use_mmsd2")
    if args.no_gpu:
        cmd.append("--no_gpu")
    subprocess.run(cmd, check=True)


def cmd_evaluate(args):
    """Launch evaluation script."""
    cmd = [
        sys.executable, "src/evaluate.py",
        f"--model={args.model}",
        f"--batch_size={args.batch_size}",
        f"--seed={args.seed}",
    ]
    if args.ckpt:
        cmd += [f"--ckpt={args.ckpt}"]
    if args.use_mmsd2:
        cmd.append("--use_mmsd2")
    if args.no_gpu:
        cmd.append("--no_gpu")
    subprocess.run(cmd, check=True)


def cmd_demo(args):
    """Quick single-sample inference demo."""
    import sys
    sys.path.insert(0, "src")

    import torch
    import json
    from pathlib import Path
    from PIL import Image
    from torchvision import transforms

    from preprocessing   import generate_sample_dataset
    from model1_cnn_lstm import build_cnn_lstm

    print("\n" + "="*55)
    print("  MEME SARCASM DETECTION – DEMO")
    print("="*55)

    # Ensure sample data exists
    if not Path("data/sample_dataset/metadata.json").exists():
        generate_sample_dataset()

    # Load a sample from the dataset
    with open("data/sample_dataset/metadata.json") as f:
        samples = json.load(f)

    sample = samples[0]
    image_path = str(Path("data/sample_dataset/images") / sample["image_file"])
    text       = sample["text"]
    true_label = sample["label"]

    # Use args if provided
    if args.image:
        image_path = args.image
    if args.text:
        text = args.text
        true_label = None

    print(f"Image : {image_path}")
    print(f"Text  : {text}")
    if true_label is not None:
        print(f"True  : {'Sarcastic' if true_label == 1 else 'Not Sarcastic'}")

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)    # (1, 3, 224, 224)

    # Dummy token IDs (for demo without tokenizer)
    ids  = torch.zeros(1, 64, dtype=torch.long)
    mask = torch.ones(1, 64, dtype=torch.long)

    # Build CNN+LSTM model (fastest for demo)
    model = build_cnn_lstm({
        "num_classes": 2, "img_embed_dim": 256, "txt_embed_dim": 256,
        "hidden_dim": 256, "dropout": 0.4, "vocab_size": 30522,
        "freeze_cnn": False,
    })

    # Load checkpoint if available
    ckpt_path = Path("outputs/checkpoints/cnn_lstm_best.pt")
    if ckpt_path.exists():
        import torch as _t
        ckpt = _t.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print("\n[Demo] Loaded trained checkpoint.")
    else:
        print("\n[Demo] No checkpoint found – using untrained model (random output).")
        print("       Run: python main.py train --model cnn_lstm  to train first.")

    model.eval()
    with torch.no_grad():
        logits = model(img_tensor, ids, mask)
        probs  = torch.softmax(logits, dim=-1)[0]
        pred   = probs.argmax().item()

    print(f"\n{'─'*40}")
    print(f"  Not Sarcastic : {probs[0].item():.4f}")
    print(f"  Sarcastic     : {probs[1].item():.4f}")
    print(f"  Prediction    : {'SARCASTIC 🎭' if pred == 1 else 'NOT SARCASTIC ✅'}")
    print(f"{'─'*40}\n")


# ──────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Meme & Sarcasm Understanding – Main Entry Point",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── setup ────────────────────────────────────
    p_setup = subparsers.add_parser("setup", help="Generate sample dataset")
    p_setup.add_argument("--n_samples", type=int, default=200)

    # ── train ────────────────────────────────────
    p_train = subparsers.add_parser("train", help="Train model(s)")
    p_train.add_argument("--model",      default="all",
                         choices=["cnn_lstm", "clip", "vbert", "all"])
    p_train.add_argument("--epochs",     type=int,   default=10)
    p_train.add_argument("--batch_size", type=int,   default=16)
    p_train.add_argument("--lr",         type=float, default=3e-4)
    p_train.add_argument("--patience",   type=int,   default=5)
    p_train.add_argument("--seed",       type=int,   default=42)
    p_train.add_argument("--use_mmsd2",  action="store_true")
    p_train.add_argument("--no_gpu",     action="store_true")

    # ── evaluate ─────────────────────────────────
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model(s)")
    p_eval.add_argument("--model",      default="all",
                        choices=["cnn_lstm", "clip", "vbert", "all"])
    p_eval.add_argument("--ckpt",       type=str, default=None)
    p_eval.add_argument("--batch_size", type=int, default=16)
    p_eval.add_argument("--seed",       type=int, default=42)
    p_eval.add_argument("--use_mmsd2",  action="store_true")
    p_eval.add_argument("--no_gpu",     action="store_true")

    # ── demo ─────────────────────────────────────
    p_demo = subparsers.add_parser("demo", help="Run inference demo")
    p_demo.add_argument("--image", type=str, default=None,
                        help="Path to a meme image")
    p_demo.add_argument("--text",  type=str, default=None,
                        help="Meme text / caption")

    args = parser.parse_args()

    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        print("\nUsage:")
        print("  python main.py setup")
        print("  python main.py train  --model all --epochs 10")
        print("  python main.py evaluate --model all")
        print("  python main.py demo   --text 'Oh wow, another bug!'")
        parser.print_help()


if __name__ == "__main__":
    main()
