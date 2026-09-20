import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .model import MeghaModel
from .config import MeghaConfig
from .tokenizer import MeghaTokenizer
import os
import re

def run_evaluation():
    print("Starting MEGHA Evaluation Phase...")
    
    # Load MEGHA
    config = MeghaConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    megha_model = MeghaModel(config).to(device)
    
    # Load the best available checkpoint (newest first)
    loaded = False
    for ckpt_name in ["checkpoints/megha_final.pt"] + \
                      [f"checkpoints/megha_level_{lvl}.pt" for lvl in range(14, -1, -1)]:
        if os.path.exists(ckpt_name):
            megha_model.load_state_dict(torch.load(ckpt_name, map_location=device))
            print(f"Loaded MEGHA from {ckpt_name}")
            loaded = True
            break
    if not loaded:
        print("CRITICAL: No checkpoint found! Evaluation will use random weights.")
        
    megha_model.eval()
    megha_tok = MeghaTokenizer(config)
    megha_tok.load("data/tokenizer.json")
    eos_id = megha_tok.get_eos_token_id()
    print(f"EOS token ID: {eos_id}")
    
    # Load Qwen (Teacher/Grader)
    print("Loading Teacher (Qwen 3B) for grading...")
    teacher_id = "Qwen/Qwen2.5-3B-Instruct"
    teacher_tok = AutoTokenizer.from_pretrained(teacher_id)
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # 10 test questions covering key curriculum levels
    test_questions = {
        "Level 1 (OS Basics)":      "What is the primary role of an operating system?",
        "Level 3 (Linux)":         "What is the command to change file permissions in Linux?",
        "Level 4 (Networking)":    "What is the purpose of DNS in computer networking?",
        "Level 5 (Databases)":     "What is the difference between a primary key and a foreign key?",
        "Level 6 (AWS)":            "What is Amazon EC2 used for?",
        "Level 7 (Docker)":         "What does a Dockerfile do?",
        "Level 8 (Kubernetes)":     "What is a Kubernetes Pod?",
        "Level 10 (Security)":      "Why should you not store AWS access keys in a public S3 bucket?",
        "Level 11 (Troubleshooting)":"If a website returns a 502 error, what could be the problem?",
        "Level 14 (CloudOps Incidents)":"How do you approach a multi-step CloudOps incident investigation?"
    }
    
    results = {}
    
    for topic, question in test_questions.items():
        print(f"\n[Testing {topic}]")
        print(f"Question: {question}")
        
        # ── 1. MEGHA generates an answer ────────────────────────────
        prompt = f"Q: {question}\nA:"
        input_ids = megha_tok.encode(prompt)
        if not input_ids:
            input_ids = [0]
            
        x = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        with torch.no_grad():
            out_ids = megha_model.generate(
                x, max_new_tokens=90, temperature=0.0, do_sample=False,
                eos_token_id=eos_id, repetition_penalty=1.05
            )
        
        # Decode ONLY the newly generated tokens (not the prompt)
        prompt_len = len(input_ids)
        new_token_ids = out_ids[0][prompt_len:].tolist()
        
        # Remove EOS token from the end if present
        if eos_id is not None and new_token_ids and new_token_ids[-1] == eos_id:
            new_token_ids = new_token_ids[:-1]
        
        megha_answer = megha_tok.decode(new_token_ids).strip()
        
        # Secondary cleanup: strip any repeated question text or A: prefix
        for junk in ["A :", "A:", question]:
            if megha_answer.startswith(junk):
                megha_answer = megha_answer[len(junk):].strip()
        
        # Cut off if a new question starts
        for stop in ["Q :", "\nQ:", " Q:"]:
            if stop in megha_answer:
                megha_answer = megha_answer.split(stop)[0].strip()
        
        # Collapse multiple spaces
        megha_answer = re.sub(r'\s+', ' ', megha_answer).strip()
        
        if not megha_answer:
            megha_answer = "[No answer generated]"
            
        print(f"MEGHA's Answer: {megha_answer}")
        
        # ── 2. Qwen grades the answer ────────────────────────────────
        grade_prompt = f"""You are an expert AI grader evaluating a small model's answer.

Question: {question}

Student's Answer: {megha_answer}

Scoring Guide:
- 80-100: Gets the core answer right (e.g. mentions 'chmod' for Linux permissions, 'translates domain names to IP addresses' for DNS, 'compute/virtual servers' for EC2, 'EXPOSE/build steps' for Docker, 'Kubernetes container group' for Pod). Full marks even if extra text is present.
- 50-70: Gets the general topic area right but misses the exact command/definition.
- 20-40: Mentions related cloud terms but inaccurate.
- 0: Completely wrong or empty.

Output ONLY a single integer score between 0 and 100."""
        
        messages = [
            {"role": "system", "content": "You are a fair, precise evaluator. Output only an integer score from 0 to 100."},
            {"role": "user", "content": grade_prompt}
        ]
        
        text = teacher_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = teacher_tok([text], return_tensors="pt").to(teacher.device)
        
        with torch.no_grad():
            gen_ids = teacher.generate(
                **model_inputs,
                max_new_tokens=5,
                do_sample=False       # greedy for consistent scoring
            )
        
        new_ids = [out[len(inp):] for inp, out in zip(model_inputs.input_ids, gen_ids)]
        score_text = teacher_tok.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
        
        # Extract first number found
        nums = re.findall(r'\d+', score_text)
        score = str(min(int(nums[0]), 100)) if nums else "0"
            
        print(f"Teacher's Grade: {score}/100")
        results[topic] = score
        
    print("\n" + "="*40)
    print("MEGHA FINAL REPORT CARD")
    print("="*40)
    total = 0
    for topic, score in results.items():
        print(f"{topic}: {score}%")
        total += int(score)
    avg = total // len(results)
    print(f"{'='*40}")
    print(f"Overall Average: {avg}%")
    print("="*40)

if __name__ == "__main__":
    run_evaluation()
