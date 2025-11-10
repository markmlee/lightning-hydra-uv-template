"""Quantitative evaluation script comparing baseline and fine-tuned models.

This script evaluates models on three metrics:
1. Perplexity - How well the model predicts the test data
2. Next Token Prediction Accuracy - How often the model predicts the correct next token
3. BLEU Score - How similar generated text is to reference text (TinyShakespeare)

Usage:
    # Evaluate fine-tuned model against baseline
    uv run src/quantitative_eval.py --checkpoint_path logs/train/runs/2025-11-07_23-52-31/checkpoints/last.ckpt

    # Use more samples for more accurate metrics
    uv run src/quantitative_eval.py --checkpoint_path CHECKPOINT_PATH --num_samples 1000

    # Generate longer sequences for BLEU
    uv run src/quantitative_eval.py --checkpoint_path CHECKPOINT_PATH --gen_length 200
"""

import argparse
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# BLEU score computation
try:
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    # Download required NLTK data
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("Warning: NLTK not available. BLEU scores will be skipped.")
    print("Install with: pip install nltk")

from src.models.smollm_lora_module import SmolLMLoRALitModule


def load_baseline_model(
    model_name: str = "HuggingFaceTB/SmolLM-135M", device: str = "cuda"
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load the baseline pretrained model."""
    print(f"\n{'=' * 70}")
    print("Loading Baseline Model")
    print(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Loaded baseline: {model_name}")
    print(f"✓ Parameters: {total_params:,}")
    print(f"✓ Device: {device}")

    return model, tokenizer


def load_finetuned_model(
    checkpoint_path: str, device: str = "cuda"
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load the fine-tuned model from checkpoint."""
    print(f"\n{'=' * 70}")
    print("Loading Fine-Tuned Model")
    print(f"{'=' * 70}")

    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path_obj}")

    print(f"Loading: {checkpoint_path_obj}")

    lit_model = SmolLMLoRALitModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )

    model = lit_model.model
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print("✓ Loaded fine-tuned checkpoint")
    print(f"✓ Total params: {total_params:,}")
    print(
        f"✓ Trainable (LoRA): {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )
    print(f"✓ Device: {device}")

    return model, tokenizer


def load_tinyshakespeare_test(
    tokenizer: PreTrainedTokenizer, max_samples: int = 500, max_length: int = 512
) -> list[Any]:
    """Load TinyShakespeare test set."""
    print(f"\n{'=' * 70}")
    print("Loading TinyShakespeare Test Set")
    print(f"{'=' * 70}")

    # Load dataset
    dataset = load_dataset("tiny_shakespeare", trust_remote_code=True)
    full_text = dataset["train"]["text"][0]  # It's all in one string

    # Tokenize
    tokenized = tokenizer(full_text, truncation=False, return_tensors="pt")
    all_input_ids = tokenized["input_ids"][0]

    # Split into sequences
    sequences = []
    for i in range(0, len(all_input_ids) - max_length, max_length):
        seq = all_input_ids[i : i + max_length]
        sequences.append(seq)

    # Use last 10% as test set (train/val/test split is typically 90/5/5 or 80/10/10)
    test_start = int(len(sequences) * 0.9)
    test_sequences = sequences[test_start : test_start + max_samples]

    print(f"✓ Total sequences: {len(sequences)}")
    print(f"✓ Test sequences: {len(test_sequences)}")
    print(f"✓ Sequence length: {max_length}")
    print(f"✓ Total test tokens: {len(test_sequences) * max_length:,}")

    return test_sequences


def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_sequences: list[Any],
    device: str = "cuda",
    batch_size: int = 8,
) -> tuple[float, float]:
    """Compute perplexity on test sequences."""
    print(f"\n{'=' * 70}")
    print("Computing Perplexity")
    print(f"{'=' * 70}")

    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(test_sequences), batch_size), desc="Perplexity"):
            batch_sequences = test_sequences[i : i + batch_size]

            # Stack into batch
            input_ids = torch.stack(batch_sequences).to(device)
            attention_mask = torch.ones_like(input_ids)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,  # For causal LM, labels = input_ids
            )

            # Accumulate loss
            loss = outputs.loss
            batch_tokens = input_ids.numel()

            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    # Compute average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    print(f"✓ Average Loss: {avg_loss:.4f}")
    print(f"✓ Perplexity: {perplexity:.2f}")

    return perplexity, avg_loss


def compute_next_token_accuracy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_sequences: list[Any],
    device: str = "cuda",
    batch_size: int = 8,
) -> float:
    """Compute next token prediction accuracy."""
    print(f"\n{'=' * 70}")
    print("Computing Next Token Prediction Accuracy")
    print(f"{'=' * 70}")

    model.eval()
    total_correct = 0
    total_predictions = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(test_sequences), batch_size), desc="Accuracy"):
            batch_sequences = test_sequences[i : i + batch_size]

            # Stack into batch
            input_ids = torch.stack(batch_sequences).to(device)
            attention_mask = torch.ones_like(input_ids)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Get predictions (logits shape: [batch, seq_len, vocab_size])
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Compare predictions with actual next tokens
            # Shift by 1: predict token i+1 from tokens 0...i
            pred_tokens = predictions[:, :-1]  # [batch, seq_len-1]
            true_tokens = input_ids[:, 1:]  # [batch, seq_len-1]

            # Count correct predictions
            correct = (pred_tokens == true_tokens).sum().item()
            total = true_tokens.numel()

            total_correct += correct
            total_predictions += total

    # Compute accuracy
    accuracy = 100 * total_correct / total_predictions

    print(f"✓ Correct predictions: {total_correct:,} / {total_predictions:,}")
    print(f"✓ Accuracy: {accuracy:.2f}%")

    return accuracy


def compute_bleu_score(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_sequences: list[Any],
    device: str = "cuda",
    num_samples: int = 15,  # Reduced for faster evaluation
    gen_length: int = 50,  # Reduced from 100 for faster evaluation
) -> float | None:
    """Compute BLEU score by generating text and comparing to reference."""
    if not BLEU_AVAILABLE:
        print("\n⚠️  BLEU score computation skipped (NLTK not available)")
        return None

    print(f"\n{'=' * 70}")
    print("Computing BLEU Score")
    print(f"{'=' * 70}")
    print(
        f"Generating {num_samples} sequences (length {gen_length}, greedy decoding for speed)..."
    )

    model.eval()
    bleu_scores = []
    smoothing = SmoothingFunction().method1

    # Use random test sequences as prompts
    import random

    sample_sequences = random.sample(
        test_sequences, min(num_samples, len(test_sequences))
    )

    with torch.no_grad():
        for seq in tqdm(sample_sequences, desc="BLEU"):
            # Use first 30 tokens as prompt (reduced for faster generation)
            prompt_length = 30
            prompt_ids = seq[:prompt_length].unsqueeze(0).to(device)

            # Generate continuation (using greedy decoding for speed)
            generated_ids = model.generate(
                prompt_ids,
                max_length=prompt_length + gen_length,
                do_sample=False,  # Greedy decoding is faster than sampling
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=1,  # No beam search for speed
            )

            # Extract only the generated part (after prompt)
            generated_ids = generated_ids[0, prompt_length:]

            # Reference is the actual continuation from the sequence
            reference_ids = seq[prompt_length : prompt_length + len(generated_ids)]

            # Decode to text
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            reference_text = tokenizer.decode(reference_ids, skip_special_tokens=True)

            # Tokenize for BLEU (word-level)
            generated_tokens = generated_text.split()
            reference_tokens = reference_text.split()

            # Compute BLEU score for this sample
            if len(generated_tokens) > 0 and len(reference_tokens) > 0:
                bleu = sentence_bleu(
                    [reference_tokens],  # Reference (list of references)
                    generated_tokens,  # Hypothesis
                    smoothing_function=smoothing,
                )
                bleu_scores.append(bleu)

    # Average BLEU score
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

    print(f"✓ Average BLEU Score: {avg_bleu:.4f}")
    print(f"✓ Samples evaluated: {len(bleu_scores)}")

    return avg_bleu


def print_comparison_table(
    baseline_results: dict[str, Any], finetuned_results: dict[str, Any]
) -> None:
    """Print a comparison table of results."""
    print(f"\n{'=' * 70}")
    print("RESULTS COMPARISON")
    print(f"{'=' * 70}\n")

    # Create table
    metrics = [
        (
            "Perplexity",
            baseline_results["perplexity"],
            finetuned_results["perplexity"],
            "lower is better",
        ),
        (
            "Average Loss",
            baseline_results["loss"],
            finetuned_results["loss"],
            "lower is better",
        ),
        (
            "Next Token Accuracy",
            baseline_results["accuracy"],
            finetuned_results["accuracy"],
            "higher is better",
        ),
    ]

    if baseline_results.get("bleu") is not None:
        metrics.append(
            (
                "BLEU Score",
                baseline_results["bleu"],
                finetuned_results["bleu"],
                "higher is better",
            )
        )

    # Print header
    print(
        f"{'Metric':<25} {'Baseline':<15} {'Fine-Tuned':<15} {'Improvement':<15} {'Note'}"
    )
    print("-" * 90)

    # Print each metric
    for metric_name, baseline_val, finetuned_val, note in metrics:
        if metric_name == "Perplexity" or metric_name == "Average Loss":
            # Lower is better
            improvement = ((baseline_val - finetuned_val) / baseline_val) * 100
            arrow = "↓" if improvement > 0 else "↑"
        else:
            # Higher is better
            improvement = ((finetuned_val - baseline_val) / baseline_val) * 100
            arrow = "↑" if improvement > 0 else "↓"

        print(
            f"{metric_name:<25} {baseline_val:<15.4f} {finetuned_val:<15.4f} {arrow} {abs(improvement):>6.2f}% {'':<6} {note}"
        )

    print()

    # Summary
    print(f"{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}\n")

    if finetuned_results["perplexity"] < baseline_results["perplexity"]:
        ppl_improvement = (
            (baseline_results["perplexity"] - finetuned_results["perplexity"])
            / baseline_results["perplexity"]
        ) * 100
        print(
            f"✅ Fine-tuned model is {ppl_improvement:.1f}% better (lower perplexity)"
        )
    else:
        print("❌ Fine-tuned model has higher perplexity (worse)")

    if finetuned_results["accuracy"] > baseline_results["accuracy"]:
        acc_improvement = finetuned_results["accuracy"] - baseline_results["accuracy"]
        print(
            f"✅ Fine-tuned model is {acc_improvement:.2f} percentage points more accurate"
        )
    else:
        print("❌ Fine-tuned model has lower accuracy")

    if finetuned_results.get("bleu") and baseline_results.get("bleu"):
        if finetuned_results["bleu"] > baseline_results["bleu"]:
            bleu_improvement = (
                (finetuned_results["bleu"] - baseline_results["bleu"])
                / baseline_results["bleu"]
            ) * 100
            print(f"✅ Fine-tuned model has {bleu_improvement:.1f}% better BLEU score")
        else:
            print("⚠️  Fine-tuned model has lower BLEU score")

    print()


def main() -> None:
    """Main entry point for quantitative evaluation."""
    parser = argparse.ArgumentParser(
        description="Quantitative evaluation of baseline vs fine-tuned models"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to fine-tuned checkpoint",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="HuggingFaceTB/SmolLM-135M",
        help="Baseline model name",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of test samples to evaluate",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for evaluation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--gen_length",
        type=int,
        default=50,
        help="Length of generated sequences for BLEU",
    )
    parser.add_argument(
        "--bleu_samples",
        type=int,
        default=15,
        help="Number of samples for BLEU computation (fewer = faster)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--skip_bleu",
        action="store_true",
        help="Skip BLEU score computation (faster)",
    )

    args = parser.parse_args()

    print(f"\n{'#' * 70}")
    print("# QUANTITATIVE MODEL EVALUATION")
    print(f"{'#' * 70}")
    print("\nEvaluating on TinyShakespeare test set")
    print(f"Samples: {args.num_samples}")
    print(f"Sequence length: {args.max_length}")
    print(f"Device: {args.device}\n")

    # Load models
    baseline_model, tokenizer = load_baseline_model(args.model_name, args.device)
    finetuned_model, _ = load_finetuned_model(args.checkpoint_path, args.device)

    # Load test data
    test_sequences = load_tinyshakespeare_test(
        tokenizer, args.num_samples, args.max_length
    )

    # Evaluate baseline model
    print(f"\n{'#' * 70}")
    print("# EVALUATING BASELINE MODEL")
    print(f"{'#' * 70}")

    baseline_ppl, baseline_loss = compute_perplexity(
        baseline_model, tokenizer, test_sequences, args.device, args.batch_size
    )
    baseline_acc = compute_next_token_accuracy(
        baseline_model, tokenizer, test_sequences, args.device, args.batch_size
    )

    if not args.skip_bleu and BLEU_AVAILABLE:
        baseline_bleu = compute_bleu_score(
            baseline_model,
            tokenizer,
            test_sequences,
            args.device,
            args.bleu_samples,
            args.gen_length,
        )
    else:
        baseline_bleu = None

    baseline_results = {
        "perplexity": baseline_ppl,
        "loss": baseline_loss,
        "accuracy": baseline_acc,
        "bleu": baseline_bleu,
    }

    # Evaluate fine-tuned model
    print(f"\n{'#' * 70}")
    print("# EVALUATING FINE-TUNED MODEL")
    print(f"{'#' * 70}")

    finetuned_ppl, finetuned_loss = compute_perplexity(
        finetuned_model, tokenizer, test_sequences, args.device, args.batch_size
    )
    finetuned_acc = compute_next_token_accuracy(
        finetuned_model, tokenizer, test_sequences, args.device, args.batch_size
    )

    if not args.skip_bleu and BLEU_AVAILABLE:
        finetuned_bleu = compute_bleu_score(
            finetuned_model,
            tokenizer,
            test_sequences,
            args.device,
            args.bleu_samples,
            args.gen_length,
        )
    else:
        finetuned_bleu = None

    finetuned_results = {
        "perplexity": finetuned_ppl,
        "loss": finetuned_loss,
        "accuracy": finetuned_acc,
        "bleu": finetuned_bleu,
    }

    # Print comparison
    print_comparison_table(baseline_results, finetuned_results)

    print(f"{'#' * 70}")
    print("# EVALUATION COMPLETE")
    print(f"{'#' * 70}\n")


if __name__ == "__main__":
    main()
