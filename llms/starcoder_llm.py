from interface.abstractmodel import AbstractModel
from transformers import AutoTokenizer, AutoModelForCausalLM

class StarCoder(AbstractModel):
    def generate_answer(self, prompt: str) -> str:
        tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase-1b")
        model = AutoModelForCausalLM.from_pretrained("bigcode/starcoderbase-1b")

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=200)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
