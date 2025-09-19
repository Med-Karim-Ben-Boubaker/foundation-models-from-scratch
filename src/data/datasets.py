from torch.utils.data import Dataset
import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


class GPTDatasetV1(Dataset):
    def __init__(self, text: str, tokenizer, max_length: int, stride: int):
        self.input_sequences, self.target_sequences = [], []

        tokenized_text = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        logger.info(
            f"Initializing GPTDatasetV1 with text length: {len(text)}, "
            f"Tokenized length: {len(tokenized_text)}, Max length: {max_length}, Stride: {stride}"
        )

        # Check if we can create any sequences
        if len(tokenized_text) <= max_length:
            logger.warning(
                f"Tokenized text length ({len(tokenized_text)}) is <= max_length ({max_length}). "
                f"No sequences can be created. Consider using a smaller max_length or longer text."
            )
            return

        for sequence_start_index in range(0, len(tokenized_text) - max_length, stride):
            input_sequence = tokenized_text[
                sequence_start_index : sequence_start_index + max_length
            ]
            target_sequence = tokenized_text[
                sequence_start_index + 1 : sequence_start_index + max_length + 1
            ]
            self.input_sequences.append(torch.tensor(input_sequence))
            self.target_sequences.append(torch.tensor(target_sequence))

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, sequence_index):
        return self.input_sequences[sequence_index], self.target_sequences[
            sequence_index
        ]
