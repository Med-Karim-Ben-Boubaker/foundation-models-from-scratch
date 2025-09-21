import os
import torch
import yaml
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from src.config import GPTConfig, TrainConfig
from src.models.gpt import GPTModel
from src.data.dataloader import create_dataloader_v1
from src.training.plotting import plot_training_history
from src.training.trainer import train
from cloud.drive_manager import DriveManager
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

    # Setup TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"llm_training_{timestamp}"
    tensorboard_log_dir = f"runs/{run_name}"
    
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    logger.info(f"TensorBoard logging to: {tensorboard_log_dir}")
    
    # Log hyperparameters to TensorBoard
    writer.add_hparams(
        {
            "learning_rate": tcfg.lr,
            "batch_size": tcfg.batch_size,
            "num_epochs": tcfg.num_epochs,
            "weight_decay": tcfg.weight_decay,
            "grad_accum_steps": tcfg.grad_accum_steps,
            "eval_freq": tcfg.eval_freq,
            "context_length": gcfg.context_length,
            "n_heads": gcfg.n_heads,
            "n_layers": gcfg.n_layers,
            "emb_dim": gcfg.emb_dim,
        },
        {}
    )

    # Check if data exists before downloading
    if not os.path.exists("data/training_data_with_special_tokens.txt"):
        logger.info("Downloading data...")
        drive = DriveManager(headless=True)
        drive.download_file(
            "training/babylm/train_10M/data/training_data_with_special_tokens.txt",
            "data/training_data_with_special_tokens.txt",
        )

    with open(
        "data/training_data_with_special_tokens.txt", "r", encoding="utf-8"
    ) as fh:
        text = fh.read()

    train_loader = create_dataloader_v1(
        text[: int(0.9 * len(text))],
        tcfg.batch_size,
        gcfg.context_length,
        gcfg.context_length // 4,
        True,
        True,
        tcfg.num_workers,
    )
    val_loader = create_dataloader_v1(
        text[int(0.9 * len(text)) :],
        tcfg.batch_size,
        gcfg.context_length,
        gcfg.context_length // 4,
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
        model, train_loader, val_loader, opt, device, tcfg, writer=writer
    )

    # plot_training_history(train_loss, val_loss, step_numbers)

    os.makedirs("artifacts", exist_ok=True)
    model_path = os.path.join("artifacts", "model_and_optimizer_2.pth")
    logger.info(f"Saving model to {model_path}")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
        },
        model_path,
    )
    logger.info(f"Model saved to {model_path}")
    
    # Close TensorBoard writer
    writer.close()
    logger.info("TensorBoard logging completed")


if __name__ == "__main__":
    main()
