import json
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Prompts for different levels of curriculum
LEVEL_PROMPTS = {
    0: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 0 - Basic English Grammar and Vocabulary.
Generate 50 simple Q&A examples covering basic sentence structure, nouns, verbs, and pronouns.
Format each example STRICTLY as: "Q: <simple question>\nA: <simple answer>".
Format the output STRICTLY as a JSON array of objects, each with a "text" field.
Example: [{"text": "Q: Is the cat sleeping?\nA: Yes, the cat is sleeping on the bed."}]
Output nothing but the JSON array. Do not include markdown blocks.""",
    
    1: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 1 - General Knowledge & Basic Reasoning.
Generate 50 Q&A examples covering numbers, comparison, time, input/output, and basic cause-effect reasoning.
Format each example STRICTLY as: "Q: <question>\nA: <clear answer>".
Format the output STRICTLY as a JSON array of objects, with each object having a "text" field.
Output nothing but the JSON array. Do not include markdown blocks.""",

    2: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 2 - Computer Fundamentals.
Generate 50 Q&A examples covering CPU, RAM, Storage (HDD vs SSD), Operating Systems (kernel, processes), and basic computing.
Format each example STRICTLY as: "Q: <question>\nA: <clear factual answer>".
Format the output STRICTLY as a JSON array of objects, with each object having a "text" field.
Output nothing but the JSON array. Do not include markdown blocks.""",

    3: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 3 - Linux Operating System.
Generate 50 Q&A examples covering Linux commands (chmod, chown, ls, grep, ps, systemctl), filesystem (/etc, /var), and file permissions.
Format each example STRICTLY as: "Q: <question>\nA: <clear factual answer>".
Format the output STRICTLY as a JSON array of objects, with each object having a "text" field.
Output nothing but the JSON array. Do not include markdown blocks.""",

    4: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 4 - Networking.
Generate Q&A examples covering TCP/IP, OSI model layers, DNS, CIDR subnetting, HTTP status codes (200, 404, 502), and common ports (22, 80, 443).
Format each example STRICTLY as: "Q: <question>\nA: <clear factual answer>".
Format the output STRICTLY as a JSON array of objects, with each object having a "text" field.
Output nothing but the JSON array. Do not include markdown blocks.""",

    5: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 5 - Cloud Computing Fundamentals.
Generate Q&A examples covering virtualization, Cloud models (IaaS, PaaS, SaaS), deployment models (public, private, hybrid), and high availability.
Format each example STRICTLY as: "Q: <question>\nA: <clear factual answer>".
Format the output STRICTLY as a JSON array of objects, with each object having a "text" field.
Output nothing but the JSON array. Do not include markdown blocks.""",

    6: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 6 - AWS Core.
Generate Q&A examples covering Amazon EC2, S3 bucket storage, IAM roles and policies, VPC, and RDS databases.
Format each example STRICTLY as: "Q: <question>\nA: <clear factual answer>".
Format the output STRICTLY as a JSON array of objects, with each object having a "text" field.
Output nothing but the JSON array. Do not include markdown blocks.""",

    7: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 7 - Docker & Containers.
Generate Q&A examples covering Docker containers, images, Dockerfile instructions (FROM, RUN, CMD, COPY), docker build, docker run, and volumes.
Format each example STRICTLY as: "Q: <question>\nA: <clear factual answer>".
Format the output STRICTLY as a JSON array of objects, with each object having a "text" field.
Output nothing but the JSON array. Do not include markdown blocks.""",

    8: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 8 - Kubernetes.
Generate Q&A examples covering K8s pods, deployments, services (ClusterIP, NodePort), Ingress, and replica sets.
Format each example STRICTLY as: "Q: <question>\nA: <clear factual answer>".
Format the output STRICTLY as a JSON array of objects, with each object having a "text" field.
Output nothing but the JSON array. Do not include markdown blocks.""",

    9: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 9 - DevOps & CI/CD.
Generate Q&A examples covering Git commands, CI/CD pipelines, and Infrastructure as Code (Terraform).
Format each example STRICTLY as: "Q: <question>\nA: <clear factual answer>".
Format the output STRICTLY as a JSON array of objects, with each object having a "text" field.
Output nothing but the JSON array. Do not include markdown blocks.""",

    10: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 10 - Cloud Security.
Generate Q&A examples covering authentication, IAM policies, why public S3 buckets are dangerous, Zero Trust, and KMS encryption keys.
Format each example STRICTLY as: "Q: <question>\nA: <clear factual answer>".
Format the output STRICTLY as a JSON array of objects, with each object having a "text" field.
Output nothing but the JSON array. Do not include markdown blocks.""",

    11: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 11 - Cloud Troubleshooting.
Generate troubleshooting Q&A examples: symptoms, diagnosis, and fix (e.g. 502 Bad Gateway cause and fix, EC2 unreachable cause and fix, S3 AccessDenied).
Format each example STRICTLY as: "Q: <troubleshooting question>\nA: <clear diagnostic and resolution steps>".
Format the output STRICTLY as a JSON array of objects, with each object having a "text" field.
Output nothing but the JSON array. Do not include markdown blocks.""",

    12: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 12 - Cloud Architecture.
Generate Q&A examples covering Highly Available designs, Load Balancer + Auto Scaling, and Serverless API architectures.
Format each example STRICTLY as: "Q: <architectural question>\nA: <clear architectural design explanation>".
Format the output STRICTLY as a JSON array of objects, with each object having a "text" field.
Output nothing but the JSON array. Do not include markdown blocks.""",

    13: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 13 - Cloud Reasoning.
Generate scenario-based Q&A examples analyzing traffic spikes, failover strategies, and database bottlenecks.
Format each example STRICTLY as: "Q: <scenario question>\nA: <logical step-by-step reasoning and solution>".
Format the output STRICTLY as a JSON array of objects, with each object having a "text" field.
Output nothing but the JSON array. Do not include markdown blocks.""",

    14: """You are an expert AI teacher generating curriculum data for a smaller language model.
Topic: Level 14 - CloudOps Multi-step Problem Solving.
Generate advanced Q&A examples showing step-by-step CloudOps problem resolution for Linux, AWS, Docker, and Kubernetes incidents.
Format each example STRICTLY as: "Q: <incident question>\nA: Identify symptoms -> Collect evidence -> Form hypothesis -> Test and Fix -> Verify.".
Format the output STRICTLY as a JSON array of objects, with each object having a "text" field.
Output nothing but the JSON array. Do not include markdown blocks."""
}

def generate_curriculum_real(level: int):
    print(f"Loading Qwen model for Level {level} curriculum generation...")
    # Upgraded Teacher to 3 Billion Parameters for much smarter data generation
    model_id = "Qwen/Qwen2.5-3B-Instruct"  
    
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
    target_examples = 300   # 300 Q&A pairs per level = 4500 total across 15 levels
    batch_size = 50         # Qwen generates 50 at a time (6 batches per level)
    
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
                
            clean_response = response.strip()
            data = None
            try:
                data = json.loads(clean_response, strict=False)
            except Exception:
                # Fallback: Robust regex extraction if LLM introduces unescaped quotes/newlines
                import re
                matches = re.findall(r'"text"\s*:\s*"(.*?)"', clean_response, re.DOTALL)
                if matches:
                    data = [{"text": m.replace('\\n', '\n').strip()} for m in matches]
                else:
                    raise
            
            # Ensure it's a list
            if isinstance(data, list):
                all_data.extend([d for d in data if isinstance(d, dict) and "text" in d and d["text"]])
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
