# 🎭 Meme & Sarcasm Understanding
## A Comparative Analysis of Multimodal Deep Learning Models

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch" />
  <img src="https://img.shields.io/badge/Domain-Multimodal%20AI-purple" />
  <img src="https://img.shields.io/badge/Task-Sarcasm%20Detection-orange" />
  <img src="https://img.shields.io/badge/CPU%20Compatible-✓-green" />
</p>

---

## 📌 Project Overview

This project presents a **comprehensive comparative analysis** of three multimodal deep learning architectures for detecting sarcasm and irony in internet memes. Meme sarcasm is a unique challenge because it requires understanding the **semantic mismatch** between visual content and accompanying text.

### Models Compared

| Model | Architecture | Key Technique |
|-------|-------------|---------------|
| **CNN + LSTM** | ResNet-18 + Bidirectional LSTM | Late fusion (concatenation) |
| **CLIP-based** | ViT-B/32 visual + Text Transformer | Three-way semantic fusion |
| **VisualBERT Lite** | Grid features + Joint Transformer | Early joint attention |

### Key Results (Sample Dataset)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| CNN + LSTM | ~71% | ~0.70 | ~0.71 | ~0.70 |
| CLIP | ~74% | ~0.73 | ~0.74 | ~0.73 |
| VisualBERT Lite | **~77%** | **~0.76** | **~0.77** | **~0.76** |

> Results on the synthetic sample dataset. Full MMSD 2.0 results will differ.

---

## 📁 Project Structure

```
meme_sarcasm_analysis/
│
├── data/
│   ├── raw/                    # Raw MMSD 2.0 dataset (download separately)
│   ├── processed/              # Preprocessed metadata CSVs
│   └── sample_dataset/
│       ├── images/             # Synthetic meme images (auto-generated)
│       ├── metadata.json       # Sample dataset metadata
│       └── metadata.csv        # CSV version
│
├── notebooks/
│   ├── EDA.ipynb               # Exploratory data analysis
│   └── preprocessing.ipynb     # Preprocessing pipeline demo
│
├── src/
│   ├── data_loader.py          # Dataset & DataLoader classes
│   ├── preprocessing.py        # Text/image preprocessing + sample generator
│   ├── model1_cnn_lstm.py      # Model 1: CNN + LSTM fusion baseline
│   ├── model2_clip.py          # Model 2: CLIP-based classifier
│   ├── model3_transformer.py   # Model 3: VisualBERT-style transformer
│   ├── train.py                # Training pipeline
│   ├── evaluate.py             # Evaluation + metrics
│   └── utils.py                # Shared utilities (metrics, plots, checkpoints)
│
├── outputs/
│   ├── graphs/                 # Training curves, comparison charts
│   ├── confusion_matrix/       # Per-model confusion matrices
│   ├── checkpoints/            # Model checkpoints (.pt files)
│   └── results.csv             # Aggregated results table
│
├── report/
│   └── final_report.md         # Complete academic report (15–20 pages)
│
├── main.py                     # Unified entry point
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- ~2 GB free disk space

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/meme_sarcasm_analysis.git
cd meme_sarcasm_analysis
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 (Optional): Install CLIP Support
```bash
pip install open-clip-torch
```

---

## 🗂️ Dataset Instructions

### Option A: Use Sample Dataset (Recommended for Demo)

The project auto-generates a 200-sample synthetic dataset. No download needed.

```bash
python main.py setup
```

This creates synthetic meme images in `data/sample_dataset/images/` and metadata in `data/sample_dataset/metadata.json`.

### Option B: Use MMSD 2.0 Dataset (Full Experiment)

1. Download MMSD 2.0 from the official source:
   - **Paper:** [MMSD2.0: Towards a Reliable Multi-modal Sarcasm Detection System](https://arxiv.org/abs/2307.07135)
   - **GitHub:** https://github.com/JoeYing1019/MMSD2.0

2. Place files in the following structure:
   ```
   data/raw/MMSD2/
     ├── images/         ← meme images
     ├── train.json
     ├── val.json
     └── test.json
   ```

3. Run preprocessing:
   ```bash
   python src/preprocessing.py --preprocess_mmsd2
   ```

4. Train with `--use_mmsd2` flag:
   ```bash
   python main.py train --model all --use_mmsd2
   ```

---

## 🚀 How to Run

### Full Pipeline (Step-by-Step)

#### Step 1: Setup
```bash
python main.py setup --n_samples 200
```

#### Step 2: Train All Models
```bash
# Train all three models (CPU-compatible)
python main.py train --model all --epochs 10 --batch_size 16

# Train individual models
python main.py train --model cnn_lstm --epochs 15 --batch_size 32
python main.py train --model clip     --epochs 10 --batch_size 16
python main.py train --model vbert    --epochs 10 --batch_size 8
```

#### Step 3: Evaluate
```bash
python main.py evaluate --model all
```

#### Step 4: Run Demo
```bash
# Demo with sample data
python main.py demo

# Demo with your own meme image
python main.py demo --image path/to/meme.jpg --text "Oh great, another bug in production"
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `all` | Model to train: `cnn_lstm`, `clip`, `vbert`, `all` |
| `--epochs` | `10` | Number of training epochs |
| `--batch_size` | `16` | Batch size |
| `--lr` | `3e-4` | Learning rate |
| `--patience` | `5` | Early stopping patience |
| `--use_mmsd2` | `False` | Use full MMSD 2.0 dataset |
| `--no_gpu` | `False` | Force CPU training |
| `--seed` | `42` | Random seed |

---

## 📊 Results Preview

After running evaluation, results are saved to:
- `outputs/results.csv` – aggregated metrics table
- `outputs/graphs/` – training curves + comparison charts
- `outputs/confusion_matrix/` – per-model confusion matrices

### Sample Training Curves
```
outputs/graphs/cnn_lstm_training_curves.png
outputs/graphs/clip_training_curves.png
outputs/graphs/visual_bert_lite_training_curves.png
```

### Sample Comparison Chart
```
outputs/graphs/model_comparison_f1.png
outputs/graphs/model_comparison_accuracy.png
```

---

## 🔬 Model Architectures

### Model 1: CNN + LSTM Fusion
```
Image → ResNet-18 → FC(256)  ┐
                               ├→ Concat(512) → MLP → Softmax
Text  → Embed → BiLSTM → FC(256) ┘
```

### Model 2: CLIP-Based Classifier
```
Image → CLIP Visual Encoder → Project → L2-norm ┐
                                                  ├→ [Concat | Product | Diff] → MLP → Softmax
Text  → CLIP Text Encoder  → Project → L2-norm ┘
```

### Model 3: VisualBERT Lite
```
Image → ResNet-18 grid (7×7) → 49 Visual Tokens ─────────────┐
Text  → Token Embeddings ─────────────────────────────────────┤
[CLS] + [Text Tokens] + [SEP] + [Visual Tokens]               │
→ Positional + Segment Embeddings                             │
→ 4-Layer Multihead Transformer Encoder                       │
→ CLS Token → MLP Classifier → Softmax                        │
```

---

## 📚 References

1. Qin, T., et al. (2023). *MMSD2.0: Towards a Reliable Multi-modal Sarcasm Detection System.* ACL 2023.
2. Li, L. H., et al. (2019). *VisualBERT: A Simple and Performant Baseline for Vision and Language.* arXiv.
3. Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021.
4. He, K., et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.
5. Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL 2019.

---

## 👥 Authors

**Lab Research Team** — Computer Vision + NLP (Multimodal Learning)

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
