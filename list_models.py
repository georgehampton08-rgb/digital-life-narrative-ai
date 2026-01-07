import google.generativeai as genai
import sys
from organizer.config import get_config, APIKeyManager

def list_models():
    config = get_config()
    manager = APIKeyManager(config.key_storage_backend, config.encrypted_key_file_path)
    api_key = manager.retrieve_key()
    if not api_key:
        print("No API key")
        return
    genai.configure(api_key=api_key)
    try:
        for m in genai.list_models():
            print(m.name)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_models()
