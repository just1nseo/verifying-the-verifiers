import os

import fire
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def maybe_merge_lora_model(model_path: str) -> str | None:
    """
    If the model is a LoRA model, merge the model and tokenizer files.
    """
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        return None

    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_name_or_path = peft_config.base_model_name_or_path
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path, trust_remote_code=True
    )

    embedding_size = base_model.get_input_embeddings().weight.shape[0]

    if len(tokenizer) > embedding_size:
        print(
            f"The vocabulary the tokenizer contains {len(tokenizer) - embedding_size} more tokens than the base model."
        )
        print("Resizing the token embeddings of the merged model...")
        base_model.resize_token_embeddings(len(tokenizer))

    # hard-coded for llama 3.1
    if embedding_size == 128256:
        base_model.resize_token_embeddings(128264)

    print("Loading the lora model...")
    lora_model = PeftModel.from_pretrained(base_model, model_path)

    print("Merging the lora modules...")
    merged_model = lora_model.merge_and_unload()
    merged_model = merged_model.to(torch.bfloat16)

    # Save the merged model to the temporary directory
    temp_dir = os.path.join(model_path, "temp")
    merged_model.save_pretrained(temp_dir)
    tokenizer.save_pretrained(temp_dir)
    return temp_dir


if __name__ == "__main__":
    fire.Fire(maybe_merge_lora_model)
