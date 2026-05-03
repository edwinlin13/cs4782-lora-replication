import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import csv
import ssl
from urllib.error import URLError
import urllib.request


SPECIAL_TOKENS = {"mr_token": "<MR>", "text_token": "<TEXT>"}
E2E_BASE_URL = "https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/"
E2E_FILES = {
    "train": "trainset.csv",
    "validation": "devset.csv",
    "test": "testset_w_refs.csv",
}


def load_e2e_dataset():
    """Load E2E NLG directly from the official CSV files."""
    dataset = {}

    for split, filename in E2E_FILES.items():
        url = E2E_BASE_URL + filename
        try:
            with urllib.request.urlopen(url) as response:
                text = response.read().decode("utf-8")
        except URLError as exc:
            if not isinstance(exc.reason, ssl.SSLCertVerificationError):
                raise
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=context) as response:
                text = response.read().decode("utf-8")

        rows = []
        reader = csv.DictReader(text.splitlines())
        for row in reader:
            rows.append({
                "meaning_representation": row["mr"],
                "human_reference": row["ref"],
            })
        dataset[split] = rows

    return dataset


def get_tokenizer():
    """Load GPT-2 tokenizer with special tokens for E2E task."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # gpt2 doesnt have a pad token by default so we use eos
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({
        "additional_special_tokens": [SPECIAL_TOKENS["mr_token"], SPECIAL_TOKENS["text_token"]]
    })
    return tokenizer


class E2EDataset(Dataset):
    """E2E NLG dataset formatted for GPT-2 autoregressive training.

    Each example is formatted as:
        <MR> meaning_representation <TEXT> reference_text <|endoftext|>
    """

    def __init__(self, hf_dataset, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        mr_token = SPECIAL_TOKENS["mr_token"]
        text_token = SPECIAL_TOKENS["text_token"]

        # format each example as a single string for autoregressive training
        for item in hf_dataset:
            prompt = f"{mr_token} {item['meaning_representation']} {text_token} {item['human_reference']}{tokenizer.eos_token}"
            encoded = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
            self.examples.append({
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, pad_token_id):
    """Pad sequences to the longest in the batch."""
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        ids = item["input_ids"]
        mask = item["attention_mask"]
        padding_length = max_len - len(ids)

        input_ids.append(ids + [pad_token_id] * padding_length)
        attention_mask.append(mask + [0] * padding_length)
        # labels are same as input_ids but -100 for padding so loss ignores them
        # took us a while to figure out the -100 thing lol
        label = ids + [-100] * padding_length
        labels.append(label)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def get_dataloaders(batch_size=8, max_length=256):
    """Load E2E NLG dataset and return train/val/test DataLoaders."""
    from functools import partial

    dataset = load_e2e_dataset()
    tokenizer = get_tokenizer()
    pad_id = tokenizer.pad_token_id

    train_ds = E2EDataset(dataset["train"], tokenizer, max_length)
    val_ds = E2EDataset(dataset["validation"], tokenizer, max_length)
    test_ds = E2EDataset(dataset["test"], tokenizer, max_length)

    collate = partial(collate_fn, pad_token_id=pad_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    return train_loader, val_loader, test_loader, tokenizer
