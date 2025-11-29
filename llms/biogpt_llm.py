from interface.abstractmodel import AbstractModel
from transformers import set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM
import torch

class BioGPT(AbstractModel):
    def generate_answer(self, prompt: str) -> str:
        tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
		
        inputs = tokenizer(prompt, return_tensors="pt")

        set_seed(42)
	
        with torch.no_grad():
            beam_output = model.generate(**inputs,
                                         min_length=100,
                                         max_length=1024,
                                         num_beans=5,
                                         early_stopping=True
                                         )

        return tokenizer.decode(beam_output[0], skip_special_tokens=True)
