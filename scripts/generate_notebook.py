import json
import os

files = [
    "megha/__init__.py",
    "megha/config.py",
    "megha/model.py",
    "megha/tokenizer.py",
    "megha/dataset.py",
    "megha/data_gen.py",
    "megha/train.py",
    "megha/evaluate.py"
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

# Construct generation strings for levels 0 to 14
gen_commands = [f"!python megha/data_gen.py --level {i} --real\n" for i in range(15)]

# Execution cell
execution_source = gen_commands + [
    "!python -m megha.tokenizer\n",
    "!python -c \"from megha.train import train_level; [train_level(i) for i in range(15)]\"\n",
    "!python -m megha.evaluate\n",
    "!cp -r checkpoints/* /kaggle/working/ || true\n"
]

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": execution_source
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
