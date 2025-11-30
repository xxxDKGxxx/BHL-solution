from interface.abstractmodel import AbstractModel
from transformers import set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM
import torch

class BioGPTForBiology(AbstractModel):
    def generate_answer(self, prompt: str) -> str:
        tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

        inputs = tokenizer(prompt, return_tensors="pt")

        set_seed(42)

        with torch.no_grad():
            beam_output = model.generate(**inputs,
                                         max_new_tokens=120,
                                         min_length=20,
                                         num_beams=5,
                                         early_stopping=True,
                                         do_sample=False,
                                         no_repeat_ngram_size=3,
                                         repetition_penalty=1.2,
                                         eos_token_id=model.config.eos_token_id,
                                         pad_token_id=model.config.eos_token_id
                                         )

        return tokenizer.decode(beam_output[0], skip_special_tokens=True)
