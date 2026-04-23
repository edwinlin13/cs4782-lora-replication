# Data

This project uses the E2E NLG dataset.

## Automatic Download

The experiment notebook loads the dataset automatically via the HuggingFace `datasets` library:

    from datasets import load_dataset
    dataset = load_dataset("e2e_nlg")

## Manual Download

Alternatively, download from: https://github.com/tuetschek/e2e-dataset

Place the CSV files in this directory:
- `trainset.csv`
- `devset.csv`
- `testset_w_refs.csv`
