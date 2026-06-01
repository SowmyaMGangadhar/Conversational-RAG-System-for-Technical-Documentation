def citation_prompt(query, context):

    prompt = f"""
        you are a technical documentation assistant

        Answer ONLY from provided context.
        Cite sources whenver possible.

        If tnformation is missing,
        say you do not know.
        _______________
        CONTEXT:
        {context}
        _______________

        QUESTION:
        {query}

        Provide:
        1. Answer
        2. Source references
        """
    return prompt
