from google import genai
from organizer.config import get_config, APIKeyManager
import sys

def test_new_sdk():
    config = get_config()
    manager = APIKeyManager(config.key_storage_backend, config.encrypted_key_file_path)
    api_key = manager.retrieve_key()
    if not api_key:
        print("No API key found.")
        return
        
    client = genai.Client(api_key=api_key)
    
    model_name = "models/gemini-1.5-flash"
    print(f"Testing {model_name} with the new google-genai SDK...")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Say 'SDK migration successful'"
        )
        print(f"SUCCESS: {response.text}")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_new_sdk()
