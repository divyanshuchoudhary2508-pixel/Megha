import json
import argparse
import os

# Qwen Curriculum Prompt Template
LEVEL_0_PROMPT = """
You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 0 - Basic English Grammar and Vocabulary.
Generate 5 simple training examples. Format the output STRICTLY as a JSON array of objects.
Each object must have a "text" field containing a grammatically perfect, simple sentence.
Example: [{"text": "The server is running."}, {"text": "Is the database active?"}]
Output nothing but the JSON array.
"""

def generate_curriculum(level: int):
    print(f"Generating curriculum for Level {level}...")
    
    # Yahan par real Kaggle environment mein Qwen (via HuggingFace API/Transformers) call hoga.
    # Abhi local testing ke liye hum API response ko simulate kar rahe hain.
    if level == 0:
        prompt = LEVEL_0_PROMPT
        print("Prompt sent to Teacher (Qwen):\n", prompt)
        
        # Simulated Qwen JSON response
        simulated_response = [
            {"text": "The computer is on."},
            {"text": "A network connects devices."},
            {"text": "She types on the keyboard."},
            {"text": "Data is stored in memory."},
            {"text": "He clicks the mouse."}
        ] * 100 # Multiply to simulate enough data for batches
    else:
        simulated_response = [{"text": f"Dummy data for level {level}."}]
        
    return simulated_response

def save_curriculum(data, level):
    # Data ko /data/ folder mein save karenge (jo gitignored hai)
    # Taki ye heavy text files GitHub par upload na hon
    os.makedirs("data", exist_ok=True)
    file_path = f"data/level_{level}_curriculum.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Curriculum saved to {file_path} successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MEGHA Curriculum using Qwen")
    parser.add_argument("--level", type=int, default=0, help="Curriculum level to generate")
    args = parser.parse_args()
    
    generated_data = generate_curriculum(args.level)
    save_curriculum(generated_data, args.level)
