import torch


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss, count = 0.0, 0
    if len(data_loader) == 0:
        return float("nan")
    for i, (x, y) in enumerate(data_loader):
        if num_batches is not None and i >= num_batches:
            break
        with torch.no_grad():
            loss = calc_loss_batch(x, y, model, device)
        total_loss += loss.item()
        count += 1
    return total_loss / max(1, count)
