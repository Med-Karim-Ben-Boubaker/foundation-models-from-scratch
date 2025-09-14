import torch
import tiktoken


def get_tokenizer(name: str = "gpt2"):
    return tiktoken.get_encoding(name)


def text_to_token_ids(text: str, tokenizer) -> torch.Tensor:
    return torch.tensor(
        tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    ).unsqueeze(0)  # add a dimension to the tensor for the batch dimension


def token_ids_to_text(token_ids: torch.Tensor, tokenizer) -> str:
    return tokenizer.decode(
        token_ids.squeeze(0).tolist()
    )  # remove the dimension from the tensor.
