import json
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Prompts for different levels of curriculum
LEVEL_PROMPTS = {
    0: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 0 - Basic English Grammar and Vocabulary.
Generate 50 simple training examples covering basic sentence structure, nouns, verbs, pronouns, and basic reasoning (e.g., 'The server is running', 'it -> EC2').
Format the output STRICTLY as a JSON array of objects, with each object having a "text" field.
Example: [{"text": "The server is running."}, {"text": "Is the database active?"}]
Output nothing but the JSON array. Do not include markdown blocks like ```json.""",
    
    1: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 1 - General Knowledge & Basic Reasoning.
Generate 50 training examples covering: Numbers (counting, comparison), Time (seconds, hours), Common Concepts (input, output, process), and Basic Reasoning (e.g. 'If a server is powered off, it cannot serve requests.').
Format the output STRICTLY as a JSON array of objects, with each object having a "text" field.
Output nothing but the JSON array. Do not include markdown blocks.""",

    2: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 2 - Computer Fundamentals.
Generate 50 training examples covering: Computer Architecture (CPU, ALU, RAM), Memory & Storage (virtual memory, HDD vs SSD), Operating Systems (kernel, system calls, threads), and Basic Programming Concepts.
Format the output STRICTLY as a JSON array of objects, with each object having a "text" field containing factual, clear statements.
Output nothing but the JSON array. Do not include markdown blocks.""",

    3: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 3 - Linux Operating System.
Generate 50 training examples covering: Linux Filesystem (/, /etc, /var), Essential Commands (ls, mkdir, grep, chmod), Processes & Services (ps, kill, systemctl), and Networking (ping, curl).
Format the output STRICTLY as a JSON array of objects, with each object having a "text" field containing factual, clear statements.
Output nothing but the JSON array. Do not include markdown blocks.""",

    4: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 4 - Networking.
Generate training examples covering: TCP/IP, OSI Model, Subnetting (CIDR), DNS, HTTP Status Codes, and common Ports (22, 80, 443).
Format the output STRICTLY as a JSON array of objects, with each object having a "text" field containing factual, clear statements.
Output nothing but the JSON array. Do not include markdown blocks.""",

    5: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 5 - Cloud Computing Fundamentals.
Generate training examples covering: Virtualization (hypervisors, VMs), Cloud Models (IaaS, PaaS, SaaS), Deployment Models (public, private, hybrid), and Cloud Characteristics (elasticity, high availability).
Format the output STRICTLY as a JSON array of objects, with each object having a "text" field containing factual, clear statements.
Output nothing but the JSON array. Do not include markdown blocks.""",

    6: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 6 - AWS Core.
Generate training examples covering: AWS EC2 (instances, AMIs), S3 (buckets, objects), IAM (roles, policies), VPC (subnets, internet gateways), and RDS.
Format the output STRICTLY as a JSON array of objects, with each object having a "text" field containing factual, clear statements.
Output nothing but the JSON array. Do not include markdown blocks.""",

    7: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 7 - Docker & Containers.
Generate training examples covering: Docker fundamentals (images, containers, Dockerfile), commands (docker build, docker run, docker ps), and Docker networking/volumes.
Format the output STRICTLY as a JSON array of objects, with each object having a "text" field containing factual, clear statements.
Output nothing but the JSON array. Do not include markdown blocks."""
}

def generate_curriculum_real(level: int):
    print(f"Loading Qwen model for Level {level} curriculum generation...")
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
    
    all_data = []
    target_examples = 1000  # We want 1000 high-quality examples per level
    batch_size = 50         # Qwen will generate 50 at a time
    
    print(f"Teacher is generating {target_examples} examples for Level {level} (in batches)...")
    
    iterations = target_examples // batch_size
    for i in range(iterations):
        print(f"Generating batch {i+1}/{iterations}...")
        try:
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=2048,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Parse JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
                
            data = json.loads(response.strip())
            
            # Ensure it's a list
            if isinstance(data, list):
                all_data.extend(data)
            elif isinstance(data, dict) and "text" in data:
                all_data.append(data)
                
            print(f"Current total for Level {level}: {len(all_data)} examples.")
            
        except Exception as e:
            print(f"Batch {i+1} failed to parse or generate, skipping. Error: {e}")
            
    print(f"Successfully generated {len(all_data)} high-quality examples for Level {level}!")
    return all_data

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
