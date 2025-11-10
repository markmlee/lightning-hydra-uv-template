"""Simple hello world example to test wandb integration."""

import wandb  # type: ignore[import-untyped]


def test_wandb_basic() -> None:
    """Basic wandb hello world test - logs a few dummy metrics."""
    # Initialize a wandb run
    wandb.init(  # type: ignore[attr-defined]
        project="LLM_finetuning",
        entity="moonrobotics",
        name="hello-world-test",
        tags=["test", "hello-world"],
        mode="online",  # Use "offline" if you want to test without internet
    )

    # Log some dummy hyperparameters
    wandb.config.update(  # type: ignore[attr-defined]
        {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
        }
    )

    # Simulate logging metrics over several steps/epochs
    for epoch in range(5):
        metrics = {
            "train/loss": 1.0 / (epoch + 1),  # Decreasing loss
            "train/acc": epoch * 0.15,  # Increasing accuracy
            "val/loss": 1.2 / (epoch + 1),
            "val/acc": epoch * 0.14,
            "epoch": epoch,
        }
        wandb.log(metrics)  # type: ignore[attr-defined]
        print(f"Epoch {epoch}: {metrics}")

    # Finish the run
    wandb.finish()  # type: ignore[attr-defined]
    print("✓ Wandb test completed successfully!")


def test_wandb_offline() -> None:
    """Test wandb in offline mode (no internet required)."""
    wandb.init(  # type: ignore[attr-defined]
        project="LLM_finetuning",
        entity="moonrobotics",
        name="hello-world-offline",
        tags=["test", "offline"],
        mode="offline",  # Offline mode - saves locally
    )

    # Log a simple metric
    for i in range(3):
        wandb.log({"test_metric": i * 2, "step": i})  # type: ignore[attr-defined]

    wandb.finish()  # type: ignore[attr-defined]
    print("✓ Offline wandb test completed successfully!")


if __name__ == "__main__":
    print("Running basic wandb test...")
    try:
        test_wandb_basic()
    except Exception as e:
        print(f"✗ Basic test failed: {e}")
        print("\nTrying offline mode instead...")
        try:
            test_wandb_offline()
        except Exception as e:
            print(f"✗ Offline test also failed: {e}")
            print("\nPlease check:")
            print("  1. Run 'wandb login' to authenticate")
            print("  2. Check your internet connection")
            print("  3. Verify entity 'moonrobotics' exists and you have access")
