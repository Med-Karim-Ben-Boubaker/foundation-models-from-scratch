import matplotlib.pyplot as plt


def plot_training_history(train_loss, val_loss, step_numbers):
    plt.figure(figsize=(12, 6))

    # Plot training loss (available at every step)
    plt.plot(step_numbers, train_loss, label="Training Loss", alpha=0.7, linewidth=1)

    # Plot validation loss only where it exists (skip None values)
    val_steps = []
    val_losses = []

    for step, loss in zip(step_numbers, val_loss):
        if loss is not None:
            val_steps.append(step)
            val_losses.append(loss)

    if val_steps:  # Only plot if we have validation data
        plt.plot(
            val_steps,
            val_losses,
            label="Validation Loss",
            marker="o",
            markersize=4,
            linewidth=2,
            alpha=0.8,
        )

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
