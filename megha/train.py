import torch
import torch.optim as optim
from .model import MeghaModel
from .config import MeghaConfig
import time

def run_dummy_test():
    print("Running Local CPU Dummy Test...")
    config = MeghaConfig()
    model = MeghaModel(config)
    
    # Calculate parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params / 1e6:.2f} M")
    
    # Dummy data: Batch of 4, sequence length of 32
    # Input tokens (idx) and shifted targets
    dummy_input = torch.randint(0, config.vocab_size, (config.batch_size, 32))
    dummy_targets = torch.randint(0, config.vocab_size, (config.batch_size, 32))
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    print("\nStarting Training Loop Test...")
    for step in range(10):
        t0 = time.time()
        
        optimizer.zero_grad()
        logits, loss = model(dummy_input, targets=dummy_targets)
        loss.backward()
        optimizer.step()
        
        dt = time.time() - t0
        print(f"Step {step} | Loss: {loss.item():.4f} | Time: {dt*1000:.2f}ms")
        
    print("\nDummy test successful! No crashes. Loss is decreasing/updating.")

if __name__ == "__main__":
    run_dummy_test()
