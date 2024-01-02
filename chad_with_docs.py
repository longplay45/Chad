import argparse
import time

from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from vectorstore import load_vectorstore

DEFAULT_TEMPLATE = """
### System:
You are a respectful and honest assistant. Your role is to answer user questions using only the provided context. If you don't know the answer, simply respond that you don't know.

### Context:
{context}

### User:
{question}

### Response:
"""


def init_template(template=DEFAULT_TEMPLATE):
    """Initialize the prompt template."""
    return template


def init_prompt():
    """Create a prompt template from the initialized template."""
    return PromptTemplate.from_template(init_template())


def init_model(name, num_gpu=0):
    """Initialize the language model with optional GPU support."""
    if num_gpu > 0:
        return Ollama(model=name, num_gpu=num_gpu)
    else:
        return Ollama(model=name)


def init_chain(prompt, llm):
    """Initialize the retrieval chain with the specified prompt and language model."""
    retriever = load_vectorstore(as_retriever=True)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def ask(query, chain):
    """Get a response from the chain for a given query."""
    return chain({"query": query})["result"]


def debug(question, model_name):
    """Debugging function to measure loading and thinking times."""
    print("-" * 50)
    print(f"loading model\t", end="")
    start_time = time.time()
    prompt = init_prompt()
    model = init_model(model_name, num_gpu=1)
    chain = init_chain(prompt, model)
    end_time = time.time()
    time_taken_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"{time_taken_ms:.2f} ms")

    print(f"thinking\t", end="")
    start_time = time.time()
    response = ask(question, chain)
    end_time = time.time()
    time_taken_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"{time_taken_ms:.2f} ms")
    print("-" * 50)
    print(f"question\t{question}\nresponse\t{response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run chain_with_rag.py with a specified question and model"
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        default="Who is Paul Graham?",
        help="Question to ask the AI",
    )
    
    parser.add_argument(
        "model_name",
        type=str,
        nargs="?",
        default="llama2",
        help="Name of the language model (e.g., llama2. try ollama list.)",
    )

    args = parser.parse_args()
    debug(args.question, args.model_name)
