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
    
    print(f"\nBatches per epoch: {total_batches}")
    print(f"Epochs: {config.epochs}")
    print(f"Total training steps: {total_batches * config.epochs}\n")
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    
    # Cosine LR scheduler for smooth convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_batches * config.epochs
    )
    
    model.train()
    global_step = 0
    for epoch in range(config.epochs):
        print(f"\n--- Epoch {epoch+1}/{config.epochs} ---")
        epoch_loss = 0.0
        
        for step, (x, y) in enumerate(dataloader):
            t0 = time.time()
            
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(x, targets=y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            dt = time.time() - t0
            epoch_loss += loss.item()
            global_step += 1
            
            if step % 20 == 0 or step == total_batches - 1:
                print(f"Step {global_step} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {dt*1000:.1f}ms")
        
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
