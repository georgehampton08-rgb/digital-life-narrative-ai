import google.generativeai as genai
from organizer.config import get_config, APIKeyManager
import sys

def test_specific_models():
    config = get_config()
    manager = APIKeyManager(config.key_storage_backend, config.encrypted_key_file_path)
    api_key = manager.retrieve_key()
    if not api_key:
        print("No API key found.")
        return
        
    genai.configure(api_key=api_key)
    
    test_cases = ["gemini-1.5-flash", "models/gemini-1.5-flash", "gemini-1.5-pro", "models/gemini-1.5-pro"]
    
    for model_name in test_cases:
        print(f"\nTesting {model_name}...")
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Hi")
            print(f"SUCCESS: {model_name} works. Response: {response.text}")
        except Exception as e:
            print(f"FAILED: {model_name} error: {e}")

if __name__ == "__main__":
    test_specific_models()
