# config.py

import configparser
import os


def load_config(file="./config.ini"):
    config = configparser.ConfigParser()
    config.read(file)
    return config


def get_logfile_path():
    config = load_config()
    path = config["general_settings"]["VECTORESTORE_DIR"]
    log_file = config["general_settings"]["VECTORESTORE_FILE"]
    return os.path.join(path, log_file)


def get_ollama_api():
    config = load_config()
    return config["general_settings"]["OLLAMA_API"]


def get_conversation_type_path():
    config = load_config()
    path = config["general_settings"]["CONVERSATION_TYPE_DIR"]
    file = config["general_settings"]["CONVERSATION_TYPE_FILE"]
    return os.path.join(path, file)
