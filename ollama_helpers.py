# ollama_helpers.property

import time

import psutil
import requests
import torch
from langchain.llms import Ollama
from config import get_ollama_api

OLLAMA_API = get_ollama_api()


def init_model(
    name,
    temperature=0,
    system_prompt=None,
    top_k=None,
    top_p=None,
    num_ctx=2048,
    num_gpu=0,
    verbose=False,
):
    """
    Initialize the Ollama model.

    See the Docs: https://api.python.langchain.com/en/latest/llms/langchain_community.llms.ollama.Ollama.html

    Args:
        name (str): Name of the model.
        num_gpu (int): Number of GPUs to use (default is 0).
        temperature (float): Temperature parameter for generation (default is 0.5).
        verbose (bool): Whether to enable verbose mode (default is False).
        num_ctx: Optional[int] = None - Sets the size of the context window used to generate the next token. (Default: 2048)
        top_k: Optional[int] = None - Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)
        top_p: Optional[int] = None - Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)

    Returns:
        Ollama: Initialized Ollama model instance.
    """

    if verbose:
        for arg, value in locals().items():
            print(f"{arg}: {value}")
    
    return Ollama(
        model=name,
        temperature=temperature,
        system=system_prompt,
        top_k=top_k,
        top_p=top_p,
        num_ctx=num_ctx,
        num_gpu=num_gpu,
        verbose=verbose,
    )


def get_timestamp_with_ms():
    """
    Get the current timestamp with milliseconds.

    Returns:
        str: Timestamp in the format "%Y-%m-%d %H:%M:%S.%f".
    """
    return time.strftime("%Y-%m-%d %H:%M:%S.%f")


def is_resource_available(url):
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.RequestException as e:
        return False


def get_local_models():
    url = f"{OLLAMA_API}/api/tags"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data["models"]]
        else:
            return f"Error: {response.status_code}"
    except requests.RequestException as e:
        return f"Error: {str(e)}"


def get_total_ram():
    # Get the system information
    system_info = psutil.virtual_memory()

    # Get the total RAM in gigabytes (GB)
    total_ram_gb = round(system_info.total / (1024**3), 2)

    return total_ram_gb


def is_mps_supported():
    if torch.backends.mps.is_available():
        return True
    else:
        return False


if __name__ == "__main__":
    # Check if the Ollama backend is available
    if is_resource_available(OLLAMA_API):
        local_models = list(get_local_models())
        print(local_models)
    else:
        print("Ollama backend is not available.")
