from interface.abstractmodel import AbstractModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

class DeepSeekCoder(AbstractModel):
    def generate_answer(self, prompt: str) -> str:
        model_name = "deepseek-ai/deepseek-math-7b-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        model.generation_config = GenerationConfig.from_pretrained(model_name)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

        text = "The integral of x^2 from 0 to 2 is"
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)

        return tokenizer.decode(outputs[0], skip_special_tokens=True)
