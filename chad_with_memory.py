import argparse
import time

from chad_conversation_types import load_conversation_type
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from ollama_helpers import init_model

# Load default conversation settings
DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_SYSTEM_PROMPT = load_conversation_type()


def init_template(system_prompt):
    """
    Initialize the template for the conversation.

    Args:
        system_prompt (str): The system prompt to use.

    Returns:
        str: The conversation template.
    """
    template = (
        system_prompt
        + """

Previous conversation:
{chat_history}

New human question: {question}

AI Response: """
    )
    return template


def init_prompt(template):
    """
    Initialize and display the prompt template.

    Args:
        template (str): The template to use for the prompt.

    Returns:
        PromptTemplate: The initialized prompt template.
    """
    
    return PromptTemplate.from_template(template)


def init_memory():
    """
    Initialize the conversation buffer memory.

    Returns:
        ConversationBufferMemory: The initialized memory.
    """
    return ConversationBufferMemory(memory_key="chat_history")


def clear_memory(memory):
    """
    Clear the memory.

    Args:
        memory (ConversationBufferMemory): The memory to clear.

    Returns:
        bool: True if the memory is cleared.
    """
    memory.clear()
    return True


def init_chain(model, temperature, system_prompt, num_gpu, verbose=False):
    """
    Initialize the language model chain.

    Args:
        model (str): The model name.
        temperature (float): The temperature setting for the model.
        system_prompt (str): The system prompt.
        num_gpu (int): Number of GPUs to use.
        verbose (bool): Whether to enable verbose logging.

    Returns:
        LLMChain: The initialized language model chain.
    """
    # Initialize the model
    model = init_model(
        name=model,
        temperature=temperature,
        system_prompt=system_prompt,
        num_gpu=num_gpu,
        verbose=verbose,
    )

    # Initialize memory, template, and prompt
    memory = init_memory()
    template = init_template(system_prompt)
    prompt = init_prompt(template)

    # Return the chain
    return LLMChain(memory=memory, prompt=prompt, llm=model, verbose=verbose)


def debug(question, model_name):
    """
    Debug function to measure loading and thinking times.

    Args:
        question (str): The question to ask the AI.
        model_name (str): The model name to use.

    Returns:
        None
    """
    print("-" * 50)
    print(f"Loading model\t", end="")
    start_time = time.time()

    # Initialize the conversation chain
    conversation = init_chain(
        model_name, DEFAULT_TEMPERATURE, DEFAULT_SYSTEM_PROMPT, num_gpu=1, verbose=True
    )

    end_time = time.time()
    print(f"Loading time: {1000 * (end_time - start_time):.2f} ms")

    print(f"Thinking...")
    start_time = time.time()
    response = conversation.run(question=question)
    end_time = time.time()

    if "i'm bob" in question:
        print(f"Thinking twice...")
        start_time = time.time()
        response = conversation.run(question="what is my name.")
        end_time = time.time()

    print("-" * 50)
    print(f"Thinking time: {1000 * (end_time - start_time):.2f} ms")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"System prompt: {DEFAULT_SYSTEM_PROMPT}")
    print(f"Human: {question}")
    print(f"AI: {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Chad Chain Memory with a specified model and question"
    )
    parser.add_argument(
        "model_name",
        type=str,
        nargs="?",
        default=DEFAULT_MODEL,
        help="Name of the language model",
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        default="i'm bob.?",
        help="Question to ask the AI",
    )

    args = parser.parse_args()
    debug(args.question, args.model_name)
