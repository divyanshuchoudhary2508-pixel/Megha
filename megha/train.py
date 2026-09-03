import torch
import torch.optim as optim
from .model import MeghaModel
from .config import MeghaConfig
from .dataset import get_dataloader
import time
import os

def train_level(level: int):
    print(f"Starting Training for Level {level}...")
    config = MeghaConfig()
    
    # Kaggle par jab GPU hoga toh yahan automatically 'cuda' select ho jayega
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MeghaModel(config).to(device)
    if level > 0:
        prev_ckpt = f"checkpoints/megha_level_{level-1}.pt"
        if os.path.exists(prev_ckpt):
            print(f"Loading previous knowledge from {prev_ckpt} for Continual Learning...")
            model.load_state_dict(torch.load(prev_ckpt, map_location=device))
        else:
            print(f"Warning: {prev_ckpt} not found! Starting from scratch...")
            
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params / 1e6:.2f} M")
    
    data_path = f"data/level_{level}_curriculum.json"
    tokenizer_path = "data/tokenizer.json"
    
    try:
        dataloader, tokenizer = get_dataloader(data_path, tokenizer_path, config)
    except FileNotFoundError as e:
        print(e)
        return
        
    print(f"Dataset loaded. Total batches per epoch: {len(dataloader)}")
    if len(dataloader) == 0:
        print("Data is too small for the batch size! Try generating more sentences in data_gen.py")
        return
        
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    model.train()
    for epoch in range(config.epochs):
        print(f"\n--- Epoch {epoch+1}/{config.epochs} ---")
        
        for step, (x, y) in enumerate(dataloader):
            t0 = time.time()
            
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(x, targets=y)
            loss.backward()
            optimizer.step()
            
            dt = time.time() - t0
            
            if step % 10 == 0 or step == len(dataloader) - 1:
                print(f"Step {step} | Loss: {loss.item():.4f} | Time: {dt*1000:.2f}ms")
                
    # Save checkpoint weights for inference later
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/megha_level_{level}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nTraining complete. Model weights saved to {checkpoint_path}")

if __name__ == "__main__":
    train_level(0)
