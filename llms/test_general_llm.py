from llms.general_llm import GeneralLLM

if __name__ == "__main__":
    llm = GeneralLLM("math")
    print(llm.generate_answer("What is pi number?"))
    llm = GeneralLLM("bio")
    print(llm.generate_answer("What is ibuprofen?"))
    llm = GeneralLLM("code")
    print(llm.generate_answer("What is python?"))
