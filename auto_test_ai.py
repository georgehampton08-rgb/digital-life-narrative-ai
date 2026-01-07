from google import genai
from organizer.config import get_config, APIKeyManager
import sys

def auto_test():
    config = get_config()
    manager = APIKeyManager(config.key_storage_backend, config.encrypted_key_file_path)
    api_key = manager.retrieve_key()
    if not api_key:
        print("No API key found.")
        return
        
    client = genai.Client(api_key=api_key)
    
    print("Listing models...")
    models = list(client.models.list())
    if not models:
        print("No models found!")
        return
        
    for m in models:
        print(f"Available model: {m.name}")
        
    target = models[0].name
    print(f"\nTesting first available model: {target}...")
    try:
        response = client.models.generate_content(
            model=target,
            contents="Hi"
        )
        print(f"SUCCESS with {target}: {response.text}")
    except Exception as e:
        print(f"FAILED with {target}: {e}")

if __name__ == "__main__":
    auto_test()
