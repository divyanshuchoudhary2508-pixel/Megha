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
    model_id = "Qwen/Qwen2.5-3B-Instruct"  
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    base_prompt = LEVEL_PROMPTS.get(level, LEVEL_PROMPTS[0])
    
    all_data = []
    target_examples = 250   # 250 Q&A pairs per level = ~3,750 total across 15 levels
    max_batches = 12
    
    print(f"Teacher is generating {target_examples} examples for Level {level} (in batches)...")
    
    sub_seeds = [
        "Focus on fundamental definitions, core concepts, and standard syntax.",
        "Focus on specific CLI flags, parameter configurations, and file paths.",
        "Focus on practical real-world scenarios, common pitfalls, and edge cases.",
        "Focus on security considerations, permissions, access controls, and best practices.",
        "Focus on error codes, log analysis, troubleshooting steps, and recovery.",
        "Focus on performance optimization, scaling, resource allocation, and automation."
    ]
    
    batch_count = 0
    while len(all_data) < target_examples and batch_count < max_batches:
        batch_count += 1
        sub_seed = sub_seeds[(batch_count - 1) % len(sub_seeds)]
        
        dynamic_prompt = f"{base_prompt}\n\nBatch {batch_count} instructions: {sub_seed}\nGenerate 30 unique Q&A examples. Output format MUST be:\nQ: <question>\nA: <answer>\n\nFormat all examples line-by-line as above."
        
        messages = [
            {"role": "system", "content": "You are an expert AI teacher generating curriculum dataset examples."},
            {"role": "user", "content": dynamic_prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=2048,
                    temperature=0.85,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            gen_tokens = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0].strip()
            
            # Primary parse: Regex extraction of Q: ... A: ... blocks
            import re
            qa_blocks = re.findall(r'(Q:\s*.*?\n\s*A:\s*.*?)(?=\n\s*Q:|\Z)', response, re.DOTALL)
            
            parsed_count = 0
            for block in qa_blocks:
                clean_block = block.strip()
                if "Q:" in clean_block and "A:" in clean_block and len(clean_block) >= 30:
                    all_data.append({"text": clean_block})
                    parsed_count += 1
                    
            # Secondary fallback: If regex found nothing, try JSON parse
            if parsed_count == 0:
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0]
                try:
                    data = json.loads(response.strip(), strict=False)
                    if isinstance(data, list):
                        for d in data:
                            if isinstance(d, dict) and "text" in d:
                                all_data.append(d)
                except Exception:
                    pass
                    
            print(f"Batch {batch_count} generated {parsed_count} examples. Total for Level {level}: {len(all_data)}/{target_examples}")
            
        except Exception as e:
            print(f"Batch {batch_count} generation error: {e}")
            
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

def is_good_example(text: str) -> bool:
    """Filter out low-quality Q&A examples before training."""
    if not text or len(text.strip()) < 30:
        return False   # Too short — probably a parse fragment
    if len(text) > 900:
        return False   # Too long — probably multiple Q&As fused together
    if text.count("Q:") > 2:
        return False   # Multiple questions jammed in one entry
    if text.count("A:") > 2:
        return False   # Multiple answers jammed in one entry
    if not ("Q:" in text and "A:" in text):
        return False   # Not Q&A format at all
    return True

def save_curriculum(data, level):
    os.makedirs("data", exist_ok=True)

    # Apply quality filter
    before = len(data)
    data = [item for item in data if is_good_example(item.get("text", ""))]
    after = len(data)
    print(f"Quality filter: {before} → {after} examples ({before - after} removed)")

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
