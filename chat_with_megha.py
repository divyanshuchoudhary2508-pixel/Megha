import torch
import os
from megha.model import MeghaModel
from megha.config import MeghaConfig
from megha.tokenizer import MeghaTokenizer

def chat():
    print("=========================================")
    print("☁️ WELCOME TO MEGHA CLOUDOPS AI ☁️")
    print("=========================================")
    print("Loading 30 Million Parameter Brain...")

    config = MeghaConfig()
    device = torch.device("cpu") # Run on CPU so it works easily on laptop
    model = MeghaModel(config).to(device)
    
    # Load the absolute latest knowledge (Level 14)
    ckpt_path = "checkpoints/megha_level_14.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found!")
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    tokenizer = MeghaTokenizer(config)
    tokenizer.load("data/tokenizer.json")
    
    print("Model Loaded Successfully! (Type 'quit' or 'exit' to stop)")
    print("=========================================\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                break
                
            prompt = f"Q: {user_input}\nA:"
            input_ids = tokenizer.encode(prompt)
            if len(input_ids) == 0:
                print("MEGHA: (I didn't understand the words...)")
                continue
                
            x = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            # Generate answer token by token
            print("MEGHA:", end=" ", flush=True)
            with torch.no_grad():
                for _ in range(40): # max 40 tokens answer
                    logits, _ = model(x)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                    x = torch.cat((x, next_token), dim=1)
                    
                    token_str = tokenizer.decode([next_token.item()])
                    print(token_str, end="", flush=True)
                    
                    # Stop if it generates a dot or newline
                    if token_str in [".", "\n", "!", "?"]:
                        break
            print("\n")
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    chat()
