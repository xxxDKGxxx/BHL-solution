from app.interface.abstractmodel import AbstractModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class QwenForIT(AbstractModel):
    def generate_answer(self, prompt: str) -> str:
        checkpoint = "Qwen/Qwen2.5-0.5B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            device_map="auto",
            dtype=torch.float16
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            min_length=20,
            temperature=0.3,
            top_p=0.8,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

        return tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
