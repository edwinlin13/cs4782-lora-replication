# LoRA Replication - CS 4782 Final Project

Replicating a key result from [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by Hu et al. We re-implement LoRA for GPT-2 on the E2E NLG data-to-text task and compare full fine-tuning with parameter-efficient LoRA adapters.

**Team:** Edwin Lin, Thomas Peng, Henry Ji  
**Course:** CS 4782, Cornell, Spring 2026

## 1. Introduction

This repository contains our CS 4782 final project re-implementation of LoRA. LoRA freezes a pretrained language model and trains small low-rank adapter matrices, reducing the number of trainable parameters while aiming to preserve full fine-tuning quality.

## 2. Chosen Result

We targeted Table 3 from the LoRA paper, specifically the GPT-2 results on the E2E NLG Challenge. That table is important because it shows LoRA matching or improving full fine-tuning on generation while training far fewer parameters.

## 3. GitHub Contents

```text
code/      LoRA, sequential LoRA, training, evaluation, notebooks, and tests
data/      Dataset README. E2E is downloaded automatically through HuggingFace datasets
results/   Metric JSON files and generated figures from our runs
poster/    Poster directory for the in-class presentation PDF
report/    Final 2-page project report PDF and LaTeX source
```

## 4. Re-implementation Details

- **Model and dataset:** GPT-2 Small on E2E NLG, where structured restaurant slot-value inputs are converted into natural language descriptions.
- **Base LoRA:** We add trainable low-rank updates to GPT-2 attention query and value projections while freezing the base model.
- **Metrics:** BLEU and ROUGE-L on generated test outputs, reported on a 0-100 scale.
- **Modifications:** We used GPT-2 Small for the main reproduction because we initially under-estimated the compute needed to exactly reproduce GPT-2 Medium/Large from the paper. We also added sequential LoRA, which grows adapter rank over training in rank-2 stages, and later ran GPT-2 Medium LoRA/SLoRA extensions in Colab.

## 5. Reproduction Steps

Clone the repository and install dependencies:

```bash
git clone https://github.com/edwinlin13/cs4782-lora-replication.git
cd cs4782-lora-replication
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Run local implementation checks:

```bash
cd code
python test_lora.py
python test_sequential_lora.py
python test_sequential_train.py
cd ..
```

To reproduce the main results, run the notebooks in `code/`:

- `experiment_colab_run.ipynb` reproduces full fine-tuning and standard LoRA ranks 2, 4, and 8.
- `sequential_experiment.ipynb` runs the sequential LoRA extension and contains the completed GPT-2 Medium Colab runs.

The notebooks download the E2E NLG dataset automatically through HuggingFace `datasets`, train GPT-2 Small variants, evaluate generated outputs, and write results to `results/metrics/` and `results/figures/`.

We recommend running the notebooks in Google Colab or on a local CUDA GPU. Our experiments used an A100 GPU. CPU execution is not practical for reproducing the reported results. The GPT-2 Medium extension is substantially slower than the GPT-2 Small runs. There are no command-line arguments for the main experiments because hyperparameters are set directly in the notebooks.

## 6. Results and Insights

| Experiment | Trainable Params | % Total | BLEU | ROUGE-L |
|---|---:|---:|---:|---:|
| Full fine-tuning | 124.44M | 100.0000 | 64.76 | 68.24 |
| LoRA r=2 | 73.7K | 0.0592 | 64.20 | 67.39 |
| LoRA r=4 | 147.5K | 0.1184 | 64.34 | 67.87 |
| LoRA r=8 | 294.9K | 0.2364 | **65.21** | 67.23 |

Our results match the paper's main pattern: LoRA stays close to full fine-tuning while training hundreds of times fewer parameters. Sequential LoRA was competitive but did not clearly beat standard LoRA r=8, suggesting that when rank is added during training matters.

Additional GPT-2 Medium extension results are tracked in `results/metrics/` and `results/metrics/sequential/`:

| Medium Experiment | Final Rank | Trainable Params | BLEU | ROUGE-L |
|---|---:|---:|---:|---:|
| LoRA r=4 | 4 | 393.2K | **65.94** | **69.25** |
| LoRA r=8 | 8 | 786.4K | 64.58 | 68.67 |
| SLoRA fixed, frozen old | 8 | 196.6K | 64.82 | 68.01 |
| SLoRA fixed, hybrid old | 8 | 196.6K | 63.96 | 67.62 |
| SLoRA fixed, unfrozen old | 8 | 196.6K | 64.56 | 68.77 |
| SLoRA plateau, frozen old | 6 | 196.6K | 64.51 | 67.18 |
| SLoRA plateau, hybrid old | 8 | 196.6K | 65.08 | 68.18 |
| SLoRA plateau, unfrozen old | 4 | 196.6K | 65.77 | 68.12 |

## 7. Conclusion

The re-implementation supports LoRA's parameter-efficiency claim at smaller scale. The main implementation challenge was correctly wrapping GPT-2's packed attention projection and verifying that base parameters stayed frozen.

## 8. References

- Hu, E. J. et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). arXiv:2106.09685, 2021.
- Novikova, J., Dusek, O., and Rieser, V. [The E2E Dataset: New Challenges for End-to-End Generation](https://arxiv.org/abs/1706.09254). arXiv:1706.09254, 2017.
- Radford, A. et al. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language-models.pdf). OpenAI, 2019.
- Wolf, T. et al. [Transformers: State-of-the-Art Natural Language Processing](https://aclanthology.org/2020.emnlp-demos.6/). EMNLP, 2020.

## 9. Acknowledgements

This project was completed for Cornell CS 4782 in Spring 2026.
