from llms.gemini_llm import GeminiLLM

llm = GeminiLLM()

def generalize_prompt(prompt: str) -> list[str, str]:
    """
    Zwraca :
    - generalized_prompt — uogólnioną wersję promptu, zachowującą sens i szczegóły
    - default_answer — bazową odpowiedź wygenerowaną na podstawie uogólnionego promptu
    """


    generalization_instruction = (
        "Please generalize the following user prompt, keeping all essential information, "
        "but rewriting it to be more universal, reusable and structured. "
        "User prompt:\n"
        f"{prompt}"
    )
    generalized_prompt = llm.generate_answer(generalization_instruction)

    answer_instruction = (
        "Using the generalized prompt below, generate a helpful default answer. "
        "The answer must:\n"
        "- include information contained in the prompt,\n"
        "- add a few extra useful remarks or clarifications,\n"
        "- remain relevant and factual.\n\n"
        f"Generalized prompt:\n{generalized_prompt}"
    )
    default_answer = llm.generate_answer(answer_instruction)

    return generalized_prompt, default_answer