import json
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
    model_checkpoint_path = "artifacts/gpt2_35m_4heads_12layers.pth"

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    gpt_config = GPTConfig(**raw["model"])

    model = GPTModel(gpt_config)
    model = model.to(device)

    model_checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(model_checkpoint["model"])
    model.eval()

    logger.info("Model loaded successfully")

    prompt = """The old man found a bottle on the beach. Inside was a message from his younger self, written decades ago and lost at sea."""

    tokenizer = get_tokenizer()
    input_token_ids = text_to_token_ids(prompt, tokenizer)
    input_token_ids = input_token_ids.to(device)

    eos_token_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    max_new_tokens = 256
    logger.info("Starting generation...")
    output_token_ids, trace_data = generate(
        model, 
        input_token_ids, 
        max_new_tokens, 
        gpt_config.context_length,
        temperature=0.1,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        eos_token_id=eos_token_id,
        min_new_tokens=1,
        trace=True,
        topk_log=10,
        verbose=False,
    )
    
    generated_text = token_ids_to_text(output_token_ids, tokenizer)
    logger.info(f"Generated text: {generated_text}")
    
    # Save trace data if available
    if trace_data:
        with open("artifacts/trace.json", "w") as f:
            json.dump(trace_data, f, indent=2)
        logger.info("Trace data saved to artifacts/trace.json")

if __name__ == "__main__":
    main()