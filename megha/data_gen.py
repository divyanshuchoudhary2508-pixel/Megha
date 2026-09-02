import json
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Prompts for different levels of curriculum
LEVEL_PROMPTS = {
    0: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 0 - Basic English Grammar and Vocabulary.
Generate 50 simple training examples. Format the output STRICTLY as a JSON array of objects.
Each object must have a "text" field containing a grammatically perfect, simple sentence.
Example: [{"text": "The server is running."}, {"text": "Is the database active?"}]
Output nothing but the JSON array. Do not include markdown blocks like ```json.""",
    
    1: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 1 - Computer Fundamentals.
Generate 50 training examples about CPU, RAM, and Storage. Format the output STRICTLY as a JSON array of objects.
Each object must have a "text" field containing a factual statement.
Example: [{"text": "RAM stands for Random Access Memory."}, {"text": "The CPU processes instructions."}]
Output nothing but the JSON array. Do not include markdown blocks."""
}

def generate_curriculum_real(level: int):
    print(f"Loading Qwen model for Level {level} curriculum generation...")
    # Kaggle par Qwen 1.5B ya 3B chal jayega easily
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"  
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    prompt = LEVEL_PROMPTS.get(level, LEVEL_PROMPTS[0])
    
    messages = [
        {"role": "system", "content": "You are a highly structured data generation AI."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    print("Teacher is generating data (this might take a minute)...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048,
        temperature=0.7
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Try parsing the JSON
    try:
        # Strip markdown formatting agar Teacher ne galti se include kar diya ho
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
            
        data = json.loads(response.strip())
        print(f"Successfully generated {len(data)} high-quality examples!")
        return data
    except json.JSONDecodeError as e:
        print("Failed to parse JSON from Qwen. Raw output:")
        print(response)
        raise e

def generate_curriculum_dummy(level: int):
    print(f"Generating DUMMY curriculum for Level {level} (Local PC Test)...")
    simulated_response = [
        {"text": "The computer is on."},
        {"text": "A network connects devices."},
        {"text": "She types on the keyboard."},
        {"text": "Data is stored in memory."},
        {"text": "He clicks the mouse."}
    ] * 100
    return simulated_response

def save_curriculum(data, level):
    os.makedirs("data", exist_ok=True)
    file_path = f"data/level_{level}_curriculum.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Curriculum saved to {file_path} successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MEGHA Curriculum using Qwen")
    parser.add_argument("--level", type=int, default=0, help="Curriculum level to generate")
    parser.add_argument("--real", action="store_true", help="Use actual HuggingFace Qwen model (requires GPU)")
    args = parser.parse_args()
    
    if args.real:
        generated_data = generate_curriculum_real(args.level)
    else:
        generated_data = generate_curriculum_dummy(args.level)
        
    save_curriculum(generated_data, args.level)
