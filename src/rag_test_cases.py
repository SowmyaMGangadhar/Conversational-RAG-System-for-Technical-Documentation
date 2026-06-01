TEST_CASES = [

    {
        "query": "What is async and await in FastAPI?",

        "ground_truth":
        """
        FastAPI uses async and await to support asynchronous programming
        using coroutines. async def functions can await non-blocking I/O
        operations while allowing the server to handle multiple requests concurrently.
        """
    },

    {
        "query": "What is quantization aware training?",

        "ground_truth":
        """
        Quantization Aware Training simulates quantization during training
        so the model adapts to lower precision operations and preserves accuracy.
        """
    },

    {
        "query": "How to execute ONNX models using ONNX Runtime?",

        "ground_truth":
        """
        ONNX Runtime executes ONNX models using an InferenceSession and NumPy inputs.
        Inputs are passed to session.run() for inference.
        """
    }
]