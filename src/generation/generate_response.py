# def generate_response(llm, prompt, temperature=0.2):
#     response = llm.invoke(prompt, temperature)
#     return response.content

def generate_response(
    llm,
    prompt,
    temperature
):

    llm.temperature = temperature

    response = llm.invoke(prompt)

    return response.content