import torch
from torch.amp import autocast, GradScaler
from src.training.evaluate import calc_loss_batch, calc_loss_loader
from src.utils.logging import get_logger

logger = get_logger(__name__)


def train(
    model,
    train_loader,
    validation_loader,
    optimizer,
    device,
    training_config,
    on_step_callback=None,
):
    is_cuda = device.type == "cuda"
    mixed_precision_scaler = GradScaler("cuda", enabled=training_config.amp and is_cuda)

    total_tokens_processed = 0
    optimizer_update_step = 0

    training_losses = []
    validation_losses = []
    step_numbers = []

    model.to(device).train()

    logger.info(f"Starting training for {training_config.num_epochs} epochs...")
    logger.info(f"Gradient accumulation steps: {training_config.grad_accum_steps}")
    logger.info(f"Mixed precision training: {training_config.amp and is_cuda}")

    for current_epoch in range(training_config.num_epochs):
        epoch_training_loss = 0.0
        epoch_batches_processed = 0

        for batch_index, (input_tokens, target_tokens) in enumerate(train_loader):
            with autocast("cuda", enabled=training_config.amp and is_cuda):
                batch_loss = calc_loss_batch(input_tokens, target_tokens, model, device)
                scaled_loss = batch_loss / training_config.grad_accum_steps

            mixed_precision_scaler.scale(scaled_loss).backward()

            is_accumulation_step_complete = (
                batch_index + 1
            ) % training_config.grad_accum_steps == 0

            if is_accumulation_step_complete:
                mixed_precision_scaler.step(optimizer)
                mixed_precision_scaler.update()

                optimizer.zero_grad(set_to_none=True)

                optimizer_update_step += 1

            batch_tokens = input_tokens.numel()
            total_tokens_processed += batch_tokens

            epoch_training_loss += scaled_loss.item() * training_config.grad_accum_steps
            epoch_batches_processed += 1

            if is_accumulation_step_complete:
                # Always record training loss and step number
                training_losses.append(scaled_loss.item())
                step_numbers.append(optimizer_update_step)

            if on_step_callback:
                on_step_callback(
                    optimizer_update_step, scaled_loss.item(), total_tokens_processed
                )

            should_evaluate = (
                is_accumulation_step_complete
                and optimizer_update_step % training_config.eval_freq == 0
                and optimizer_update_step > 0
            )

            if should_evaluate:
                logger.info(f"Evaluating at step {optimizer_update_step}...")

                model.eval()
                with torch.no_grad():
                    current_validation_loss = calc_loss_loader(
                        validation_loader,
                        model,
                        device,
                        num_batches=training_config.eval_iter,
                    )

                model.train()

                validation_losses.append(current_validation_loss)

                logger.info(
                    f"Step {optimizer_update_step:>6} | "
                    f"Train Loss: {scaled_loss.item():.4f} | "
                    f"Val Loss: {current_validation_loss:.4f} | "
                    f"Tokens: {total_tokens_processed:>8}"
                )
            elif is_accumulation_step_complete:
                # Add None for validation loss when not evaluating but at gradient step
                validation_losses.append(None)

        average_epoch_loss = epoch_training_loss / max(1, epoch_batches_processed)

        model.eval()
        with torch.no_grad():
            final_validation_loss = calc_loss_loader(
                validation_loader, model, device, num_batches=training_config.eval_iter
            )
        model.train()

        logger.info(
            f"Epoch {current_epoch + 1}/{training_config.num_epochs} completed | "
            f"Avg Train Loss: {average_epoch_loss:.4f} | "
            f"Final Val Loss: {final_validation_loss:.4f} | "
            f"Total Steps: {optimizer_update_step}"
        )

    logger.info("Training completed!")
    return (
        optimizer_update_step,
        total_tokens_processed,
        training_losses,
        validation_losses,
        step_numbers,
    )
