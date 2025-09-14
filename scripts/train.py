import os
import torch
import yaml
from src.config import GPTConfig, TrainConfig
from src.models.gpt import GPTModel
from src.data.dataloader import create_dataloader_v1
from src.training.plotting import plot_training_history
from src.training.trainer import train
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cfg_path = os.environ.get("CFG", "configs/gpt_124m.yaml")
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    gcfg = GPTConfig(**raw["model"])
    tcfg = TrainConfig(**raw["train"])

    device = torch.device(tcfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    with open("data/the-verdict.txt", "r", encoding="utf-8") as fh:
        text = fh.read()

    train_loader = create_dataloader_v1(
        text[: int(0.9 * len(text))],
        tcfg.batch_size,
        gcfg.context_length,
        gcfg.context_length,
        True,
        True,
        tcfg.num_workers,
    )
    val_loader = create_dataloader_v1(
        text[int(0.9 * len(text)) :],
        tcfg.batch_size,
        gcfg.context_length,
        gcfg.context_length,
        False,
        False,
        tcfg.num_workers,
    )

    model = GPTModel(gcfg)
    model = model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay
    )

    logger.info("Starting training...")
    _, _, train_loss, val_loss, step_numbers = train(
        model, train_loader, val_loader, opt, device, tcfg
    )
    
    logger.debug(f"Train loss: {train_loss}")
    logger.debug(f"Val loss: {val_loss}")
    logger.debug(f"Step numbers: {step_numbers}")

    plot_training_history(train_loss, val_loss, step_numbers)
    
    os.makedirs("artifacts", exist_ok=True)
    model_path = os.path.join("artifacts", "model_and_optimizer.pth")
    logger.info(f"Saving model to {model_path}")
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
    }, model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
