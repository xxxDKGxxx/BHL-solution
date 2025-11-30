from interface.abstractmodel import AbstractModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
import torch

class DeepSeekCoderForMathematics(AbstractModel):
    def generate_answer(self, prompt: str) -> str:
        model_name = "deepseek-ai/deepseek-math-7b-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model.generation_config = GenerationConfig.from_pretrained(model_name)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
                                **inputs.to(model.device),
                                max_new_tokens=200,
                                no_repeat_ngram_size=3,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.eos_token_id
                                )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)
