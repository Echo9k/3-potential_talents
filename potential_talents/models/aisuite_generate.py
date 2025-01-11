import os
import aisuite as ai


def aisuite_generate(messages, model="openai:o1-mini", max_tokens=1500, temperature=0.0):
    """
    Generate a response using AISuite's chat completion API.
    
    Args:
        messages (list): List of message dicts for the model.
        model (str): Model identifier to use.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.

    Returns:
        str: Generated response text.
    """
    client = ai.Client(
        provider_configs={
            "groq": {
                "api_key": os.environ["GROQ_API_KEY"]
            },
            "huggingface": {
                "api_key": os.environ["HUGGINGFACE_TOKEN"]
            },
            "openai": {
                "api_key": os.environ["OPENAI_API_KEY"]
            }
        }
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_tokens,
        # temperature=temperature,  # Uncomment if supported
    )
    return response.choices[0].message.content