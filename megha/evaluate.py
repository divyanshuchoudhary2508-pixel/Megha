import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .model import MeghaModel
from .config import MeghaConfig
from .tokenizer import MeghaTokenizer
import os

def run_evaluation():
    print("Starting MEGHA Evaluation Phase (Level 15)...")
    
    # Load MEGHA
    config = MeghaConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    megha_model = MeghaModel(config).to(device)
    
    # Load the best available checkpoint (Level 14 first, then fallback)
    loaded = False
    for lvl in range(14, -1, -1):
        ckpt_path = f"checkpoints/megha_level_{lvl}.pt"
        if os.path.exists(ckpt_path):
            megha_model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"Loaded MEGHA from {ckpt_path} (Level {lvl})")
            loaded = True
            break
    if not loaded:
        print("CRITICAL: No checkpoint found at all! Evaluation will use random weights.")
        
    megha_model.eval()
    megha_tok = MeghaTokenizer(config)
    megha_tok.load("data/tokenizer.json")
    
    # Load Qwen (Teacher)
    print("Loading Teacher (Qwen 3B) for grading...")
    teacher_id = "Qwen/Qwen2.5-3B-Instruct"
    teacher_tok = AutoTokenizer.from_pretrained(teacher_id)
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # 5 Sample Questions covering the syllabus
    test_questions = {
        "Level 3 (Linux)": "What is the command to change file permissions in Linux?",
        "Level 6 (AWS)": "What is Amazon EC2 used for?",
        "Level 7 (Docker)": "What does a Dockerfile do?",
        "Level 10 (Security)": "Why should you not store AWS access keys in a public S3 bucket?",
        "Level 11 (Troubleshooting)": "If a website returns a 502 error, what could be the problem?"
    }
    
    results = {}
    
    for topic, question in test_questions.items():
        print(f"\n[Testing {topic}] Question: {question}")
        
        # 1. MEGHA generates an answer
        prompt = f"Q: {question}\nA:"
        input_ids = megha_tok.encode(prompt)
        
        # If tokenizer returns empty or very short, pad it safely
        if len(input_ids) == 0:
            input_ids = [0]
            
        x = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        # Generate tokens using temperature=0.5 (focused, not random)
        with torch.no_grad():
            out_ids = megha_model.generate(x, max_new_tokens=50, temperature=0.5, top_k=40)
                    
        full_decoded = megha_tok.decode(out_ids[0].tolist())
        # Clean up the output string
        megha_answer = full_decoded.replace(prompt, "").replace("<|endoftext|>", "").strip()
        # Cut off at next question if it rambles
        if "Q:" in megha_answer:
            megha_answer = megha_answer.split("Q:")[0].strip()
            
        print(f"MEGHA's Answer: {megha_answer}")
        
        # 2. Qwen grades the answer
        grade_prompt = f"""You are grading an AI student's answer.
Question: {question}
Student's Answer: {megha_answer}
Rate the student's answer out of 100 based on accuracy and conceptual relevance. If partially correct or relevant keywords are used, award partial marks (e.g., 40 to 80). If completely wrong or total gibberish, award 0.
Output ONLY the numeric score (e.g., 75)."""
        
        messages = [
            {"role": "system", "content": "You are a strict grader. Output only a number between 0 and 100."},
            {"role": "user", "content": grade_prompt}
        ]
        
        text = teacher_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = teacher_tok([text], return_tensors="pt").to(teacher.device)
        
        generated_ids = teacher.generate(
            **model_inputs,
            max_new_tokens=10,
            temperature=0.1
        )
        
        generated_ids = [
            out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)
        ]
        
        score_text = teacher_tok.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        # Ensure score is just digits
        score_text = ''.join(filter(str.isdigit, score_text))
        if not score_text: score_text = "0"
            
        print(f"Teacher's Grade: {score_text}/100")
        results[topic] = score_text
        
    print("\n" + "="*40)
    print("MEGHA FINAL REPORT CARD")
    print("="*40)
    for topic, score in results.items():
        print(f"{topic}: {score}%")
    print("="*40)

if __name__ == "__main__":
    run_evaluation()
