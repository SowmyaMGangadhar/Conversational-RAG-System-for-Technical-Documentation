def guardrail_prompt(query, context):

    prompt = f"""
        you are a safe RAG assistant.

        STRICT RULES:
        - Use ONLY retrieved context
        - Do NOT hallucinate
        - Do NOT invent APIs
        - DO NOT generate unsupported claims

        if the answer is unavailable:
        say:
        "Documentation does not contain this information."

        _______________
        CONTEXT:
        {context}
        _______________

        QUESTION:
        {query}        
        """
    return prompt
