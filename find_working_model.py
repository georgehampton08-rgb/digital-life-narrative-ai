from google import genai
from organizer.config import get_config, APIKeyManager
import sys

def find_working_model():
    config = get_config()
    manager = APIKeyManager(config.key_storage_backend, config.encrypted_key_file_path)
    api_key = manager.retrieve_key()
    if not api_key:
        print("No API key found.")
        return
        
    client = genai.Client(api_key=api_key)
    
    print("Listing and testing models...")
    try:
        models = list(client.models.list())
        if not models:
            print("No models found!")
            return
            
        for m in models:
            # Skip embeddings for content generation test
            if "embedding" in m.name:
                continue
                
            print(f"Testing {m.name}...")
            try:
                response = client.models.generate_content(
                    model=m.name,
                    contents="Hi, say 'OK'"
                )
                print(f" SUCCESS: {m.name} works! Response: {response.text}")
                # We found a winner
                return m.name
            except Exception as e:
                print(f" FAILED: {m.name} error: {e}")
                
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    winner = find_working_model()
    if winner:
        print(f"\nFinal Verified Model: {winner}")
    else:
        print("\nNo working model found.")
