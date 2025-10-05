"""Main training script for sleep classification."""

from src import SimpleTrainer, get_config


if __name__ == "__main__":
    config = get_config()
    trainer = SimpleTrainer(config=config)
    trainer.run()
