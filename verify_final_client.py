from organizer.ai.client import get_client
import logging

# Setup logging to see what's happening
logging.basicConfig(level=logging.DEBUG)

def verify_client():
    print("Initializing AIClient with new SDK and verified model...")
    try:
        client = get_client()
        print(f"Client initialized with model: {client.model_name}")
        
        print("Testing generation...")
        response = client.generate("Hi, say 'verified'")
        print(f"SUCCESS: {response.text}")
        
        print("Testing JSON generation...")
        json_resp = client.generate_json("Return a JSON with key 'status' and value 'ok'")
        print(f"JSON SUCCESS: {json_resp}")
        
    except Exception as e:
        print(f"VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_client()
