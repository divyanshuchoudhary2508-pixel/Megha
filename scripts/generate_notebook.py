import json
import os

files = [
    "megha/__init__.py",
    "megha/config.py",
    "megha/model.py",
    "megha/tokenizer.py",
    "megha/dataset.py",
    "megha/data_gen.py",
    "megha/train.py"
]

cells = []

# Cell to create directories and install dependencies
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!mkdir -p megha data checkpoints\n",
        "!pip install transformers tokenizers accelerate huggingface_hub\n"
    ]
})

# Embed each python file as a cell using %%writefile
for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    
    source = [f"%%writefile {fpath}\n"]
    source.extend([line + "\n" for line in content.split("\n")])
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    })

# Execution cell
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!python megha/data_gen.py --level 0 --real\n",
        "!python -m megha.tokenizer\n",
        "!python -m megha.train\n",
        "!cp -r checkpoints/* /kaggle/working/ || true\n"
    ]
})

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("scripts/kaggle_runner/runner.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully with embedded files!")
