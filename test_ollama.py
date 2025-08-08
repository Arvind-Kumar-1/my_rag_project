# test_ollama.py
import ollama

print("--- Starting Ollama Python Connection Test ---")

try:
    print("Attempting to connect to Ollama and ask a simple question...")
    print("This may take a moment as the model loads into memory...")
    
    # This is the simplest possible call to the library.
    # It will test the connection and the model itself.
    response = ollama.chat(
        model='llama3',
        messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    )
    
    print("\n--- TEST SUCCEEDED ---")
    print("Successfully received a response from Ollama.")
    print("\nLlama 3 says:")
    print(response['message']['content'])

except Exception as e:
    print("\n--- TEST FAILED ---")
    print("Could not get a response from the Ollama library, even though the service is running.")
    print(f"Error: {e}")
    print("\nThis suggests an issue with your firewall, antivirus, or a resource problem (like not enough RAM).")