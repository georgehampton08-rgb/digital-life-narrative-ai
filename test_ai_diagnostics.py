import google.generativeai as genai
import os
import sys

def test_models():
    from organizer.config import get_config, APIKeyManager
    
    config = get_config()
    manager = APIKeyManager(config.key_storage_backend, config.encrypted_key_file_path)
    api_key = manager.retrieve_key()
    if not api_key:
        print("Error: Could not retrieve API key from storage.")
        sys.exit(1)
        
    genai.configure(api_key=api_key)
    
    print("Listing available models...")
    try:
        models = genai.list_models()
        model_list = list(models)
        print(f"Found {len(model_list)} models:")
        for m in model_list:
            print(f"- {m.name} (Methods: {m.supported_generation_methods})")
            
        print("\nTesting gemini-1.5-flash...")
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello, respond with 'OK'")
        print(f"Response: {response.text}")
        
    except Exception as e:
        print(f"Error during model listing or testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_models()
