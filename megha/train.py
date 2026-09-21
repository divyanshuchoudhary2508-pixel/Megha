import torch
import torch.optim as optim
from .model import MeghaModel
from .config import MeghaConfig
from .dataset import get_combined_dataloader
import time
import os

def train_all():
    """
    MIXED TRAINING: Train ONE model on ALL 15 levels' data shuffled together.
    This eliminates Catastrophic Forgetting completely.
    """
    print("=" * 50)
    print("MEGHA MIXED TRAINING — All Levels Combined")
    print("=" * 50)
    
    config = MeghaConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MeghaModel(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params / 1e6:.2f} M\n")
    
    tokenizer_path = "data/tokenizer.json"
    
    try:
        dataloader, tokenizer = get_combined_dataloader(tokenizer_path, config)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return
    
    total_batches = len(dataloader)
    if total_batches == 0:
        print("ERROR: Dataloader has 0 batches. Check your data files.")
        return
    
    grad_accum_steps = getattr(config, 'grad_accum_steps', 2)
    warmup_steps = getattr(config, 'warmup_steps', 150)
    total_steps = (total_batches // grad_accum_steps) * config.epochs
    
    print(f"\nBatches per epoch: {total_batches}")
    print(f"Gradient Accumulation Steps: {grad_accum_steps}")
    print(f"Epochs: {config.epochs}")
    print(f"Total optimization steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}\n")
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.05)
    
    def get_lr(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
        
    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)
    
    model.train()
    global_step = 0
    optimizer.zero_grad()
    
    for epoch in range(config.epochs):
        print(f"\n--- Epoch {epoch+1}/{config.epochs} ---")
        epoch_loss = 0.0
        accum_loss = 0.0
        
        for step, (x, y) in enumerate(dataloader):
            t0 = time.time()
            x, y = x.to(device), y.to(device)
            
            logits, loss = model(x, targets=y)
            loss = loss / grad_accum_steps
            loss.backward()
            accum_loss += loss.item() * grad_accum_steps
            
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                dt = time.time() - t0
                if global_step % 20 == 0 or (step + 1) == total_batches:
                    curr_lr = optimizer.param_groups[0]['lr']
                    print(f"Step {global_step}/{total_steps} | Loss: {accum_loss:.4f} | LR: {curr_lr:.2e} | Time: {dt*1000:.1f}ms")
                accum_loss = 0.0
                
            epoch_loss += loss.item() * grad_accum_steps
        
        avg_loss = epoch_loss / total_batches
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
    
    # Save the final unified checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    final_path = "checkpoints/megha_final.pt"
    torch.save(model.state_dict(), final_path)
    
    # Also save as level_14 for backwards compatibility with evaluate.py fallback
    torch.save(model.state_dict(), "checkpoints/megha_level_14.pt")
    
    print(f"\n{'='*50}")
    print(f"Training complete! Final model saved to {final_path}")
    print(f"Total steps trained: {global_step}")
    print(f"{'='*50}")


# Keep old function for backward compatibility
def train_level(level: int):
    """Deprecated: use train_all() instead."""
    print(f"Note: train_level() is deprecated. Use train_all() for better results.")
    train_all()


if __name__ == "__main__":
    train_all()
