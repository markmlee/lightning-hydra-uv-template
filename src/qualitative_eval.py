"""Qualitative evaluation script for comparing baseline and fine-tuned models.

This script loads both the baseline SmolLM-135M model and a fine-tuned checkpoint,
then generates text from both models given sample prompts to compare their outputs.

Usage:
    # Interactive mode with LoRA checkpoint (default)
    uv run src/qualitative_eval.py --checkpoint_path logs/train/runs/2025-11-09_15-21-37/checkpoints/last.ckpt

    # Interactive mode with full fine-tuned checkpoint (non-LoRA)
    uv run src/qualitative_eval.py --checkpoint_path logs/train/runs/2025-11-07_20-46-13/checkpoints/last.ckpt --no_lora

    # With predefined Shakespeare prompts
    uv run src/qualitative_eval.py --checkpoint_path CHECKPOINT_PATH --shakespeare

    # Custom prompt
    uv run src/qualitative_eval.py --checkpoint_path CHECKPOINT_PATH --prompt "To be or not to be"
"""

import argparse
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# Import your fine-tuned model classes
from src.models.smollm_lora_module import SmolLMLoRALitModule
from src.models.smollm_module import SmolLMLitModule


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

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ Loaded baseline model: {model_name}")
    print(f"✓ Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✓ Device: {device}")

    return model, tokenizer


def load_finetuned_model(
    checkpoint_path: str, device: str = "cuda", use_lora: bool = True
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load the fine-tuned model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load model on (cuda/cpu)
        use_lora: If True, load as LoRA model. If False, load as full fine-tuned model.

    Returns:
        Tuple of (model, tokenizer)

    """
    print(f"\n{'=' * 70}")
    print(f"Loading Fine-Tuned Model ({'LoRA' if use_lora else 'Full Fine-tuning'})")
    print(f"{'=' * 70}")

    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path_obj}")

    print(f"Loading checkpoint: {checkpoint_path_obj}")

    # Load appropriate module type
    if use_lora:
        lit_model = SmolLMLoRALitModule.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
        )
    else:
        lit_model = SmolLMLitModule.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
        )

    # Extract the HuggingFace model
    model = lit_model.model
    model = model.to(device)
    model.eval()

    # Use same tokenizer as baseline
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Print model info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print("✓ Loaded fine-tuned model from checkpoint")
    print(f"✓ Total parameters: {total_params:,}")
    if use_lora:
        print(f"✓ Trainable parameters (LoRA): {trainable_params:,}")
        print(f"✓ Trainable %: {100 * trainable_params / total_params:.2f}%")
    else:
        print("✓ Fine-tuning type: Full model fine-tuning")
    print(f"✓ Device: {device}")

    return model, tokenizer


def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    num_return_sequences: int = 1,
    device: str = "cuda",
) -> list[str]:
    """Generate text from a model given a prompt."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_texts = [
        tokenizer.decode(output, skip_special_tokens=True) for output in outputs
    ]

    return generated_texts


def compare_models(
    baseline_model: PreTrainedModel,
    finetuned_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.8,
    device: str = "cuda",
) -> None:
    """Compare text generation from baseline and fine-tuned models."""
    print(f"\n{'=' * 70}")
    print(f"Prompt: '{prompt}'")
    print(f"{'=' * 70}")

    # Generate from baseline
    print("\n[Baseline Model Output]")
    print("-" * 70)
    baseline_outputs = generate_text(
        baseline_model,
        tokenizer,
        prompt,
        max_length=max_length,
        temperature=temperature,
        device=device,
    )
    for text in baseline_outputs:
        print(f"{text}")

    # Generate from fine-tuned
    print("\n[Fine-Tuned Model Output]")
    print("-" * 70)
    finetuned_outputs = generate_text(
        finetuned_model,
        tokenizer,
        prompt,
        max_length=max_length,
        temperature=temperature,
        device=device,
    )
    for text in finetuned_outputs:
        print(f"{text}")

    print()


def get_shakespeare_prompts() -> list[str]:
    """Return a list of Shakespeare-style prompts for testing."""
    return [
        "To be or not to be,",
        "O Romeo, Romeo,",
        "All the world's a stage,",
        "What light through yonder window breaks?",
        "Now is the winter of our discontent",
        "Friends, Romans, countrymen,",
        "The quality of mercy is not strained,",
        "Once more unto the breach,",
    ]


def interactive_mode(
    baseline_model: PreTrainedModel,
    finetuned_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: str,
) -> None:
    """Interactive mode for custom prompts."""
    print(f"\n{'=' * 70}")
    print("Interactive Mode")
    print(f"{'=' * 70}")
    print("\nEnter prompts to compare models. Type 'quit' or 'exit' to stop.")
    print("You can also adjust generation parameters:")
    print("  - Type 'temp X' to set temperature (e.g., 'temp 0.7')")
    print("  - Type 'length X' to set max_length (e.g., 'length 150')")
    print()

    temperature = 0.8
    max_length = 100

    while True:
        try:
            prompt = input("\nPrompt> ").strip()

            if not prompt:
                continue

            if prompt.lower() in ["quit", "exit", "q"]:
                print("\nExiting interactive mode.")
                break

            # Check for parameter adjustments
            if prompt.lower().startswith("temp "):
                try:
                    temperature = float(prompt.split()[1])
                    print(f"✓ Temperature set to {temperature}")
                    continue
                except (IndexError, ValueError):
                    print("✗ Invalid temperature. Use: temp 0.8")
                    continue

            if prompt.lower().startswith("length "):
                try:
                    max_length = int(prompt.split()[1])
                    print(f"✓ Max length set to {max_length}")
                    continue
                except (IndexError, ValueError):
                    print("✗ Invalid length. Use: length 100")
                    continue

            # Generate comparison
            compare_models(
                baseline_model,
                finetuned_model,
                tokenizer,
                prompt,
                max_length=max_length,
                temperature=temperature,
                device=device,
            )

        except KeyboardInterrupt:
            print("\n\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")


def main() -> None:
    """Main entry point for qualitative evaluation."""
    parser = argparse.ArgumentParser(
        description="Compare baseline and fine-tuned models"
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
        "--prompt",
        type=str,
        help="Single prompt to test (if not provided, uses interactive mode)",
    )
    parser.add_argument(
        "--shakespeare",
        action="store_true",
        help="Use predefined Shakespeare prompts",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Generation temperature (0.0-2.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Load checkpoint as full fine-tuned model (not LoRA)",
    )

    args = parser.parse_args()

    print(f"\n{'#' * 70}")
    print("# Qualitative Model Comparison")
    print(f"{'#' * 70}")

    # Load models
    baseline_model, tokenizer = load_baseline_model(args.model_name, args.device)
    finetuned_model, _ = load_finetuned_model(
        args.checkpoint_path, args.device, use_lora=not args.no_lora
    )

    # Determine mode
    if args.prompt:
        # Single prompt mode
        compare_models(
            baseline_model,
            finetuned_model,
            tokenizer,
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            device=args.device,
        )

    elif args.shakespeare:
        # Shakespeare prompts mode
        print(f"\n{'=' * 70}")
        print("Testing with Shakespeare-style prompts")
        print(f"{'=' * 70}")

        for prompt in get_shakespeare_prompts():
            compare_models(
                baseline_model,
                finetuned_model,
                tokenizer,
                prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                device=args.device,
            )

    else:
        # Interactive mode
        interactive_mode(baseline_model, finetuned_model, tokenizer, args.device)

    print(f"\n{'#' * 70}")
    print("# Evaluation Complete")
    print(f"{'#' * 70}\n")


if __name__ == "__main__":
    main()
