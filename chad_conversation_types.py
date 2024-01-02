# chad_conversation_types.py

import os
import pandas as pd

# Import a function from the 'config' module to get the path for the conversation data.
from config import get_conversation_type_path

# Get the full path for the pickle file where data will be stored.
PKL_FULL_PATH = get_conversation_type_path()

# Define system prompts for different types of data.
conversation_systemprompt = """You are a nice chatbot having a conversation with a human."""
documents_systemprompt = """
### System:
You are a respectful and honest assistant. Your role is to answer user questions using only the provided context. If you don't know the answer, simply respond that you don't know.

### Context:
{context}

### User:
{question}

### Response:
"""
images_systemprompt = """"""

# Function to create a DataFrame with predefined setup values.
def create_dataframe():
    CONVERSATION_SETUP = dict(
        type        =["conversation",   "documents",        "images"],
        model       =["llama2:latest",  "llama2:latest",    "backllama:latest"],
        temperature =[0.5,              0,                  0],
        system_prompt=[
            conversation_systemprompt,
            documents_systemprompt,
            images_systemprompt,
        ],
    )
    return pd.DataFrame.from_dict(CONVERSATION_SETUP)

# Function to load a DataFrame from a pickle file.
def load_dataframe(full_path):
    return pd.read_pickle(full_path)

# Function to save a DataFrame to a pickle file.
def save_dataframe(df, full_path):
    df.to_pickle(full_path)

# Function to load a specific type of data from the DataFrame.
def load_conversation_type(type='conversation'):
    """
    Load a specific type of data from the DataFrame.

    Parameters:
    - type (str): The type of data to load. Options are 'conversation', 'documents', 'images'.

    Returns:
    - pd.Series: The specified type of data as a Pandas Series.
    - pd.DataFrame: The entire DataFrame if the type is not specified or invalid.
    """
    df = load_dataframe(PKL_FULL_PATH)
    
    if type:
        selected_rows = df.loc[df['type'] == type]
        model = selected_rows['model'][0]
        temperature = selected_rows['temperature'][0]
        system_prompt = selected_rows['system_prompt'][0]
        return model, temperature, system_prompt
    
    # Return the entire DataFrame if the type is not specified or invalid.
    #return df


# Debug function to create, save, and print the DataFrame.
def debug():
    df = create_dataframe()
    save_dataframe(df, PKL_FULL_PATH)
    print(df)
    print(f'Pickel file saved to {PKL_FULL_PATH}.')

# Execute the 'main' function if this script is run directly.
if __name__ == "__main__":
    debug()
