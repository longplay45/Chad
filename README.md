# > Chad 👨*.

_***Hi, my name is Chad. You can ask me anything about the world, programming, your documents or even images...and I am work in progress... ;-).**_

![Screenshot of Chad](data/screenshot_002.png)

This is a simple web-ui running local LLMs like Lama2, Mistral, etc. via [**Ollama**](https://ollama.ai) on your local machine. It is written in python using the fantastic [**Ollama**](https://ollama.ai) backend with [**LangChain**](https://www.langchain.com) and [**Streamlit**](https://streamlit.io) for the WebUI.

## Features

- Runs 100% on your local machine
- Conversation Types
  - Conversations
  - Documents (RAG)
  - Images (Multi-modal)
- Supports multiple, even unsencored LLMs via Ollama or Huggingface-Hub. 
- Build in converstion memory
- Upload your own PDFs or images to have a conversation with your data
- Simple UI built with Streamlit

## Hardware requirements

- Apple Silicon M1+
- 16 GB RAM
- python 3.10+
- [miniconda](https://formulae.brew.sh/cask/miniconda)
- [**Ollama**](https://ollama.ai)

## Screenshots

### Conversation with memory

![Screenshot of Chad](data/screenshot_001.png)

### Conversation with images

![Screenshot of Chad](data/screenshot_002.png)

### Conversation with documents

![Screenshot of Chad](data/screenshot_003.png)

### Settings

![Screenshot of Chad](data/screenshot_004.png)


## Installation

### Install Ollama

First visit their website and [download](https://ollama.ai/download) 🦙 **Ollama**.

### Download models

```bash
ollama pull llama2
ollama pull bakllama
```

### Clone this repository and cd into project directory.

```sh
git clone https://github.com/netzwerkerei/Chad.git
cd Chad
```

### Create, activate your virtual Chad evironment and install all dependencies via conda

Use the command below to create a virtual environment and install all necessary dependencies.

```sh
conda env create -f environment.yml
conda activate Chad
```

### Run the app

```sh
streamlit run Chad.py
```

### Create and configure your own model and chain settings


- Open Notebook for already installed models `./chad_settings.ipynb`
  - Change default system prompts, tempretures and models per chain

- or get others via huggingface-hub

```sh
pip install huggingface-hub

huggingface-cli download 
  TheBloke/MistralLite-7B-GGUF 
  mistrallite.Q4_K_M.gguf 
  --local-dir downloads 
  --local-dir-use-symlinks False
```

- Create and edit `./Modelfile`

```sh
FROM ./downloads/mistrallite.Q4_K_M.gguf
SYSTEM ...
```

- Create and run model

```sh
ollama create Chad -f Modelfile

ollama run Chad "What is your name?"
```

## License

Copyleft & -right [netzwerkerei@gmail.com](netzwerkerei@gmail.com) 2024.
