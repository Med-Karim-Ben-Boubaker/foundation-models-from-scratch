import torch


def generate_text_simple(model, input_token_ids, max_new_tokens, context_size):
    """
    Generate new text tokens using a simple greedy decoding strategy.

    Args:
        model: The trained language model
        input_token_ids: Starting sequence of token IDs (batch_size, seq_len)
        max_new_tokens: Number of new tokens to generate
        context_size: Maximum context window size for the model

    Returns:
        Complete sequence including original input and generated tokens
    """
    for _ in range(max_new_tokens):
        # Extract only the last context_size tokens to stay within model's context window
        context_tokens = input_token_ids[:, -context_size:]

        # Disable gradient computation for inference (faster and uses less memory)
        with torch.no_grad():
            # Get model predictions for the context tokens
            logits = model(context_tokens)[
                :, -1, :
            ]  # Only use the last position's logits

            # Select the most likely next token using greedy decoding (argmax)
            next_token_id = torch.argmax(
                torch.softmax(logits, dim=-1), dim=-1, keepdim=True
            )

            # Append the new token to the sequence
            input_token_ids = torch.cat((input_token_ids, next_token_id), dim=1)

    return input_token_ids
