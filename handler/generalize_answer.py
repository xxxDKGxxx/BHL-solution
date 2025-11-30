from interface.abstractmodel import AbstractModel

def generalize_prompt(llm: AbstractModel, prompt: str) -> tuple[str, str]:
    """
    Returns:
    - generalized_prompt: a concise, generalized version of the prompt that still keeps the original details
    - default_answer: a short, factual answer based on the generalized prompt
    """

    # Include original details in the generalization
    generalization_instruction = (
        "Keep the new prompt simple. Rewrite the following user prompt to make it more general, "
        "while keeping all the key details intact. "
        "The new prompt should explicitly request the original information, "
        "and also ask for a few additional contextual details that could improve the response.\n"
        f"Original Prompt:\n{prompt}"
    )

    generalized_prompt = llm.generate_answer(generalization_instruction)

    # Short, factual answer
    answer_instruction = (
        "Using the generalized prompt below, generate a concise and factual answer. "
        "Include the details from the original prompt exactly, and add a few extra relevant facts or clarifications.\n"
        f"Generalized prompt:\n{generalized_prompt}"
    )
    default_answer = llm.generate_answer(answer_instruction)

    return generalized_prompt, default_answer
