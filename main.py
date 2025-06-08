import json
import requests
from typing import Dict, List
import os

def get_embedding(text: str) -> List[float]:
    """Get embedding for a given text using the local API."""
    url = "http://127.0.0.1:1234/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "text-embedding-granite",
        "input": text
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

def process_messages(input_file: str, output_file: str):
    """Process messages from input JSON file and save embeddings to output file."""
    # Read input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        chat_data = json.load(f)
    
    # Get messages array from the chat data
    messages = chat_data.get('messages', [])
    
    # Process each message and get embeddings
    results = []
    total_messages = len(messages)
    processed_count = 0
    
    for message in messages:
        # Skip if not a message type or if text is empty
        if message.get('type') != 'message' or not message.get('text'):
            continue
            
        message_id = message.get('id')
        message_text = message.get('text', '')
        
        if message_id and message_text:
            try:
                embedding = get_embedding(message_text)
                results.append({
                    'id': message_id,
                    'embedding': embedding
                })
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count}/{total_messages} messages...")
            except Exception as e:
                print(f"Error processing message {message_id}: {str(e)}")
    
    # Save results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nTotal messages processed: {processed_count}")
    print(f"Total messages skipped: {total_messages - processed_count}")

def main():
    input_file = "result.json"  # Input JSON file with messages
    output_file = "embeddings.json"  # Output file for embeddings
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return
    
    try:
        process_messages(input_file, output_file)
        print(f"Successfully processed messages and saved embeddings to '{output_file}'")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
