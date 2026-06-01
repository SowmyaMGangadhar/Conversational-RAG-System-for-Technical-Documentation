def structured_prompt(query, context):

    prompt = f"""
        Answer in JSON format.

        Format:
        {{
        "answer": "...",
        "source": "...",
        "confidence": "high/medium/low"
        }}

        _______________
        CONTEXT:
        {context}
        _______________

        QUESTION:
        {query}
        
        """
    return prompt
