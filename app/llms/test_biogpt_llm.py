from app.llms.biogpt_llm import BioGPTForBiology

llm = BioGPTForBiology()

print(llm.generate_answer("What is ibuprofen?"))