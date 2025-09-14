from torch.utils.data import DataLoader
import tiktoken
from src.data.datasets import GPTDatasetV1
from src.utils.logging import get_logger

logger = get_logger(__name__)

def create_dataloader_v1(
    text, batch_size, max_length, stride, shuffle, drop_last, num_workers
):
    logger.info(f"Creating dataloader with text length: {len(text)}, Max length: {max_length}, Stride: {stride}")
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
