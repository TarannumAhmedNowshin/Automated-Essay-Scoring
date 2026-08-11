# Automated Essay Scoring for Bangla

Fine-tuning transformer encoders and large language models (LLMs) to grade
university-level **Bangla essays** on a 0–10 scale. This repository contains the
model code from our undergraduate thesis, *"Automated Essay Scoring System for
Bangla Language"* (Department of Computer Science and Engineering, BRAC
University, 2024).

Each model is framed as a **regression** task: given a `Question` and a student
`Answer`, predict the human-assigned `Marks`. We compare traditional BERT-style
encoders against decoder-only LLMs (run with 4-bit quantization on a single GPU).

📄 The full write-up is available in [thesis-report.pdf](thesis-report.pdf).

## Repository structure

```
Automated-Essay-Scoring/
├── transformermodels/            # Encoder models (BERT-family)
│   ├── banglabert-small.ipynb    # Bangla BERT — small dataset (~325)
│   ├── banglabert-large.ipynb    # Bangla BERT — large dataset (~1,408)
│   ├── indicbert-small.ipynb     # IndicBERT (AI4Bharat) — small
│   ├── indicbert-large.ipynb     # IndicBERT (AI4Bharat) — large
│   ├── muril-small.ipynb         # MuRIL base  — small
│   └── muril-large.ipynb         # MuRIL large — large
├── llmmodels/                    # Decoder-only LLMs (4-bit quantized)
│   ├── falcon-1b.ipynb           # Falcon3-1B-Instruct
│   ├── falcon-7b.ipynb           # Falcon-7B-Instruct
│   └── llama-8b.ipynb            # Llama-3.1-8B
├── data/                    # Place your CSVs here (not tracked — see data/README.md)
├── requirements.txt
└── LICENSE
```

## Models

| Notebook | Base model | Hugging Face ID |
|----------|-----------|-----------------|
| Bangla BERT | Sagor Sarker Bangla BERT | `sagorsarker/bangla-bert-base` |
| IndicBERT | AI4Bharat IndicBERT | `ai4bharat/indic-bert` |
| MuRIL (base) | Google MuRIL | `google/muril-base-cased` |
| MuRIL (large) | Google MuRIL | `google/muril-large-cased` |
| Falcon 1B | TII Falcon 3 | `tiiuae/Falcon3-1B-Instruct` |
| Falcon 7B | TII Falcon | `tiiuae/falcon-7b-instruct` |
| Llama 3.1 8B | Meta Llama 3.1 | `meta-llama/Llama-3.1-8B` |

## Results

Models were evaluated with **RMSE**, **MAE**, and **R²**. Lower RMSE/MAE and
higher R² are better. The tables below report **test-set** performance; full
train/validation/test breakdowns are in the thesis report.

### Traditional encoder models (larger dataset)

| Model | RMSE ↓ | MAE ↓ | R² ↑ |
|-------|:------:|:-----:|:----:|
| **Bangla BERT** | 0.8078 | 0.5215 | 0.2950 |
| IndicBERT (AI4Bharat) | 0.7237 | 0.5047 | 0.2527 |
| MuRIL | 0.8222 | 0.6642 | 0.2694 |

Bangla BERT fit the training data best (train R² ≈ **0.96**), confirming these
encoders learn the structural signals of Bangla essays well.

### Large language models (larger dataset)

| Model | RMSE ↓ | MAE ↓ | R² ↑ |
|-------|:------:|:-----:|:----:|
| Falcon 1B Instruct | 0.9401 | 0.6391 | 0.0452 |
| Falcon 7B Instruct | 0.9391 | 0.6510 | 0.0471 |
| Llama 3.1 8B | 1.0059 | 0.7616 | -0.0934 |

**Takeaway:** the smaller, task-specific encoder models outperformed the much
larger LLMs on this low-resource Bangla scoring task. The LLMs — even the 8B
model — struggled to generalize, largely due to limited Bangla training data and
the fact that they are built for text *generation* rather than regression.

## Dataset

The essay dataset is **not** published here (it is private student data). The
notebooks read CSVs from the `data/` folder — see [data/README.md](data/README.md)
for the expected file names and columns (`Question`, `Answer`, `Marks`).

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install -r requirements.txt
```

The LLM notebooks additionally require `bitsandbytes` for 4-bit quantization,
which needs an NVIDIA GPU with CUDA.

> **Tested with** Python 3.10–3.11 and `transformers < 4.46`. Newer `transformers`
> renamed `evaluation_strategy` → `eval_strategy`; update that argument if you use
> a newer version. See [requirements.txt](requirements.txt) for the full notes.

The LLM notebooks download gated/large models and read your token from an
environment variable — **never hardcode it**:

```bash
# Linux/macOS
export HF_TOKEN=your_token_here

# Windows (PowerShell)
setx HF_TOKEN "your_token_here"
```

`meta-llama/Llama-3.1-8B` is a gated model; request access on Hugging Face first.

## Running

1. Put your CSVs in `data/` (`essays_small.csv`, `essays_large.csv`).
2. Open a notebook and select a Python kernel with the requirements installed.
3. Run the cells top to bottom.

The encoder notebooks run on CPU or a modest GPU. The LLM notebooks were trained
on a **16 GB GPU (RTX 4080 Super)** using 4-bit quantization (QLoRA-style) to fit
the 1B–8B models in memory.

## Run in Colab

No local GPU? Open a notebook in Google Colab, set the runtime to **GPU**
(required for the LLM notebooks), and upload your CSVs to a `data/` folder in the
session.

| Notebook | Open |
|----------|------|
| Bangla BERT — small | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TarannumAhmedNowshin/Automated-Essay-Scoring/blob/main/transformermodels/banglabert-small.ipynb) |
| Bangla BERT — large | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TarannumAhmedNowshin/Automated-Essay-Scoring/blob/main/transformermodels/banglabert-large.ipynb) |
| IndicBERT — small | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TarannumAhmedNowshin/Automated-Essay-Scoring/blob/main/transformermodels/indicbert-small.ipynb) |
| IndicBERT — large | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TarannumAhmedNowshin/Automated-Essay-Scoring/blob/main/transformermodels/indicbert-large.ipynb) |
| MuRIL — small | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TarannumAhmedNowshin/Automated-Essay-Scoring/blob/main/transformermodels/muril-small.ipynb) |
| MuRIL — large | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TarannumAhmedNowshin/Automated-Essay-Scoring/blob/main/transformermodels/muril-large.ipynb) |
| Falcon 1B | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TarannumAhmedNowshin/Automated-Essay-Scoring/blob/main/llmmodels/falcon-1b.ipynb) |
| Falcon 7B | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TarannumAhmedNowshin/Automated-Essay-Scoring/blob/main/llmmodels/falcon-7b.ipynb) |
| Llama 3.1 8B | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TarannumAhmedNowshin/Automated-Essay-Scoring/blob/main/llmmodels/llama-8b.ipynb) |

## Reproducing the results

1. Install dependencies from [requirements.txt](requirements.txt) (mind the
   `transformers < 4.46` cap).
2. Place `essays_small.csv` / `essays_large.csv` in `data/` — see
   [data/README.md](data/README.md) for the expected columns.
3. Run the relevant notebook top to bottom. Each one prints **RMSE / MAE / R²**
   for the train, validation, and test splits and renders the diagnostic plots.

The numbers in the results tables come from single runs on our hardware; exact
values will vary slightly on re-runs (see **Limitations**).

## Limitations

- **Fine-tuned weights are not released** — only the training/evaluation code is
  provided, so reproducing requires the datasets and (for the LLMs) a CUDA GPU.
- **Splits are not fully deterministic.** The Bangla BERT and LLM notebooks fix
  the split seed (`random_state=42`), but the IndicBERT and MuRIL notebooks call
  `datasets.train_test_split` without a seed, so their splits — and therefore
  their metrics — differ between runs.
- **No global training seed** is set, so results are not bit-for-bit reproducible.
- The dataset is small and university-specific; treat the scores as indicative
  rather than production-grade.

## Authors

- Tarannum Ahmed Nowshin
- Tahiya Mysara Yusha
- Moinuddin Zubair

Supervised by Dr. Farig Yousuf Sadeque, BRAC University.

## Citation

If you use this work, please cite it (GitHub's "Cite this repository" uses
[CITATION.cff](CITATION.cff)), or with BibTeX:

```bibtex
@misc{aes_bangla_2024,
  title  = {Automated Essay Scoring System for Bangla Language},
  author = {Nowshin, Tarannum Ahmed and Yusha, Tahiya Mysara and Zubair, Moinuddin},
  year   = {2024},
  note   = {B.Sc. thesis, BRAC University},
  url    = {https://github.com/TarannumAhmedNowshin/Automated-Essay-Scoring}
}
```

## License

Released under the [MIT License](LICENSE).