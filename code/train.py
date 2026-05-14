import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import nltk
from rouge_score import rouge_scorer
from utils import TrainingLogger, save_checkpoint, count_parameters, save_metrics

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch=None, num_epochs=None):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    num_batches = 0

    desc = "Training"
    if epoch is not None and num_epochs is not None:
        desc = f"Training {epoch}/{num_epochs}"

    progress = _progress(dataloader, desc=desc, leave=False)
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        if tqdm is not None:
            progress.set_postfix(loss=f"{total_loss / num_batches:.4f}")
        if num_batches == 1 or num_batches % 100 == 0 or num_batches == len(dataloader):
            print(
                f"    {desc}: batch {num_batches}/{len(dataloader)} "
                f"| avg loss {total_loss / num_batches:.4f}",
                flush=True,
            )

    return total_loss / num_batches


@torch.no_grad()
def validate(model, dataloader, device, epoch=None, num_epochs=None):
    """Compute average loss on a validation/test set."""
    model.eval()
    total_loss = 0
    num_batches = 0

    desc = "Validation"
    if epoch is not None and num_epochs is not None:
        desc = f"Validation {epoch}/{num_epochs}"

    progress = _progress(dataloader, desc=desc, leave=False)
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        num_batches += 1
        if tqdm is not None:
            progress.set_postfix(loss=f"{total_loss / num_batches:.4f}")
        if num_batches == 1 or num_batches % 100 == 0 or num_batches == len(dataloader):
            print(
                f"    {desc}: batch {num_batches}/{len(dataloader)} "
                f"| avg loss {total_loss / num_batches:.4f}",
                flush=True,
            )

    return total_loss / num_batches


@torch.no_grad()
def generate_texts(model, dataset_hf, tokenizer, device, max_new_tokens=128, num_beams=10):
    """Generate text for each unique MR in the dataset."""
    model.eval()
    mr_token = "<MR>"
    text_token = "<TEXT>"

    # get unique MRs so we dont generate duplicates
    unique_mrs = list(set(item["meaning_representation"] for item in dataset_hf))
    results = []

    for idx, mr in enumerate(_progress(unique_mrs, desc="Generating", leave=False), start=1):
        prompt = f"{mr_token} {mr} {text_token}"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.pad_token_id,
        )

        # extract just the generated part after <TEXT>
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        if text_token in full_text:
            generated = full_text.split(text_token, 1)[1]
            generated = generated.replace(tokenizer.eos_token, "").strip()
        else:
            generated = full_text

        results.append((mr, generated))
        if idx == 1 or idx % 100 == 0 or idx == len(unique_mrs):
            print(f"    Generating: {idx}/{len(unique_mrs)}", flush=True)

    return results


def compute_metrics(generated_results, dataset_hf):
    """Compute BLEU and ROUGE-L against all references."""
    # build a map from MR -> list of reference texts
    mr_to_refs = {}
    for item in dataset_hf:
        mr = item["meaning_representation"]
        ref = item["human_reference"]
        if mr not in mr_to_refs:
            mr_to_refs[mr] = []
        mr_to_refs[mr].append(ref)

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    all_references = []
    all_hypotheses = []
    rouge_l_scores = []

    for mr, generated in generated_results:
        refs = mr_to_refs.get(mr, [])
        if not refs:
            continue
        all_references.append([nltk.word_tokenize(r.lower()) for r in refs])
        all_hypotheses.append(nltk.word_tokenize(generated.lower()))
        # rouge-l: take the best score across references
        rouge_l_scores.append(max(
            scorer.score(ref, generated)["rougeL"].fmeasure
            for ref in refs
        ))

    # corpus bleu handles multiple references per hypothesis
    bleu = nltk.translate.bleu_score.corpus_bleu(all_references, all_hypotheses)
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0

    return {"bleu": bleu, "rouge_l": avg_rouge_l}


def run_experiment(
    model,
    train_loader,
    val_loader,
    test_loader,
    test_dataset_hf,
    tokenizer,
    device,
    num_epochs=5,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=0,
    experiment_name="experiment",
    checkpoint_dir="checkpoints",
    results_dir="../results/metrics",
):
    """Run a full training + evaluation experiment."""
    model = model.to(device)
    param_info = count_parameters(model)
    print(f"[{experiment_name}] Trainable params: {param_info['trainable']:,} "
          f"/ {param_info['total']:,} ({param_info['percentage']:.4f}%)")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # linear warmup from basically 0 to lr
    scheduler = None
    if warmup_steps > 0:
        scheduler = LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
        )

    logger = TrainingLogger()
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        logger.start_epoch()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch=epoch + 1, num_epochs=num_epochs,
        )
        val_loss = validate(
            model, val_loader, device,
            epoch=epoch + 1, num_epochs=num_epochs,
        )
        elapsed = logger.end_epoch(train_loss, val_loss)

        print(f"  Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} "
              f"| Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                f"{checkpoint_dir}/{experiment_name}_best.pt"
            )

    print(f"  Generating on test set...")
    generated = generate_texts(model, test_dataset_hf, tokenizer, device)
    metrics = compute_metrics(generated, test_dataset_hf)
    print(f"  BLEU: {metrics['bleu']:.4f} | ROUGE-L: {metrics['rouge_l']:.4f}")

    results = {
        "experiment_name": experiment_name,
        "status": "complete",
        "params": param_info,
        "training_history": logger.get_history(),
        "test_metrics": metrics,
    }

    save_metrics(results, f"{results_dir}/{experiment_name}.json")
    return results
