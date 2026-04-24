
from unsloth import FastLanguageModel
import torch

def load_model_and_tokenizer(max_seq_length=2048, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=128,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    return model, tokenizer

