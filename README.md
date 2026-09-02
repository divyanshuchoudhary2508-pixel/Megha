# MEGHA: A Resource-Efficient Self-Improving Language Model

MEGHA is a compact language model (30-50M parameters) developed from scratch to investigate teacher-guided iterative learning for CloudOps and Computing knowledge.

## Project Structure
- `megha/`: Core source code (model, tokenizer, training loop).
- `data/`: Local data directory (ignored by git).
- `checkpoints/`: Model weights (ignored by git).
- `scripts/`: Utility scripts for running on Kaggle and syncing.

## Workflow
1. Develop and test locally (CPU/Dummy data).
2. Push to GitHub.
3. Kaggle notebook clones the repo and runs training.
4. Pull checkpoints back to local.
