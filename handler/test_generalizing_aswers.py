from generalize_answer import generalize_prompt

if __name__ == '__main__':
    prompt = 'Provide Cauchy\'s definition of convergence'
    new_prompt, answer = generalize_prompt(prompt)

    print(f"Original prompt:\n {prompt}")

    print(f"new_prompt:\n{new_prompt}")
    print(f"answer:\n{answer}")