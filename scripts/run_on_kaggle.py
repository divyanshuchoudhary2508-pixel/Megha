import os
import subprocess
import argparse

def push_to_kaggle():
    print("Packaging local code into Kaggle Notebook...")
    subprocess.run(["python", "scripts/generate_notebook.py"], check=True)
    
    print("Pushing notebook to Kaggle to start training...")
    # Navigate to the kaggle_runner folder where metadata exists
    os.chdir("scripts/kaggle_runner")
    subprocess.run(["kaggle", "kernels", "push"])
    print("Kaggle job submitted! You can check progress on Kaggle website.")

def pull_checkpoints(username):
    print("Pulling output checkpoints from Kaggle...")
    os.makedirs("checkpoints", exist_ok=True)
    kernel_id = f"{username}/megha-training-runner"
    subprocess.run(["kaggle", "kernels", "output", kernel_id, "-p", "checkpoints"])
    print(f"Checkpoints downloaded successfully to local /checkpoints folder!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="Push and run notebook on Kaggle")
    parser.add_argument("--pull", action="store_true", help="Pull checkpoints from Kaggle")
    parser.add_argument("--user", type=str, help="Your Kaggle username (required for pull)")
    args = parser.parse_args()

    if args.push:
        push_to_kaggle()
    if args.pull:
        if not args.user:
            print("Error: Please provide --user KAGGLE_USERNAME for pulling checkpoints.")
        else:
            pull_checkpoints(args.user)
