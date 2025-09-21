import torch
import yaml
from src.config import GPTConfig
from src.data.tokenizer import text_to_token_ids, get_tokenizer, token_ids_to_text
from src.models.gpt import GPTModel
from src.utils.logging import get_logger
from src.training.generate import generate

logger = get_logger(__name__)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    config_path = "configs/gpt_124m.yaml"
    model_checkpoint_path = "artifacts/model_and_optimizer.pth"

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    gpt_config = GPTConfig(**raw["model"])

    model = GPTModel(gpt_config)
    model = model.to(device)

    model_checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(model_checkpoint["model"])
    model.eval()

    logger.info("Model loaded successfully")

    prompt = "USA is a"

    tokenizer = get_tokenizer()
    input_token_ids = text_to_token_ids(prompt, tokenizer)
    input_token_ids = input_token_ids.to(device)

    # Get EOS token ID for early stopping
    eos_token_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    max_new_tokens = 100
    output_token_ids = generate(
        model, 
        input_token_ids, 
        max_new_tokens, 
        gpt_config.context_length,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        eos_token_id=eos_token_id,
        min_new_tokens=3,
    )
    
    logger.info(f"Generated text: {token_ids_to_text(output_token_ids, tokenizer)}")

if __name__ == "__main__":
    main()