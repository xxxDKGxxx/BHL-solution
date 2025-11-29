from prompts_classification.fact_or_generative_classifier import FactOrGenerativeClassifier

model = FactOrGenerativeClassifier()

while True:
    prompt = input("Enter a prompt: ")
    is_generative = model.predict(prompt)
    print(f"{prompt}: is generative? {is_generative}")