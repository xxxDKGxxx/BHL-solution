from interface.abstractmodel import AbstractModel
from transformers import AutoTokenizer, AutoModelForCausalLM

class StarCoderForIT(AbstractModel):
    def generate_answer(self, prompt: str) -> str:
        tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase-1b")
        model = AutoModelForCausalLM.from_pretrained("bigcode/starcoderbase-1b")

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
                                **inputs,
                                max_new_tokens=200,
                                no_repeat_ngram_size=3,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.eos_token_id
                                )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
