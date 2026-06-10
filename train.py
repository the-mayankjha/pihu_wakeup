import os
import sys
import subprocess

def main():
    print("="*60)
    print("openWakeWord Local Training Setup")
    print("="*60)
    print("\nTraining an openWakeWord model locally from scratch requires generating")
    print("synthetic data using Text-to-Speech (Piper) and training a logistic")
    print("regression model. This requires specific system dependencies.\n")
    
    # Check if MacOS and suggest brew install
    if sys.platform == "darwin":
        print("Since you are on macOS, you will need 'espeak-ng' installed:")
        print("  brew install espeak-ng\n")
    else:
        print("Make sure you have 'espeak-ng' installed on your system.\n")

    repo_dir = "openWakeWord_repo"
    
    if not os.path.exists(repo_dir):
        print("Cloning the official openWakeWord repository to access training scripts...")
        try:
            subprocess.run(["git", "clone", "https://github.com/dscripka/openWakeWord.git", repo_dir], check=True)
            print("Repository cloned successfully.\n")
        except subprocess.CalledProcessError:
            print("Error: Failed to clone the repository. Ensure git is installed.")
            sys.exit(1)
    else:
        print(f"Directory '{repo_dir}' already exists. Skipping clone.\n")

    print("Next Steps for Local Training:")
    print("1. Navigate to the cloned repository:")
    print(f"   cd {repo_dir}")
    print("\n2. Install the specific training requirements (recommend using a virtual environment):")
    print("   pip install -r requirements.txt")
    print("   pip install -e .")
    print("\n3. Use their automated script to train your models. Run:")
    print("   # Standard spellings")
    print("   python -m openwakeword.train --target_word \"pihu\" --output_dir ../models/")
    print("   python -m openwakeword.train --target_word \"hi pihu\" --output_dir ../models/")
    print("   python -m openwakeword.train --target_word \"hey pihu\" --output_dir ../models/")
    print("\n   # Phonetic variations (since Pihu is a non-English word, Piper TTS might mispronounce it)")
    print("   python -m openwakeword.train --target_word \"pee who\" --output_dir ../models/")
    print("   python -m openwakeword.train --target_word \"peehoo\" --output_dir ../models/")
    print("   python -m openwakeword.train --target_word \"piwho\" --output_dir ../models/")
    print("   python -m openwakeword.train --target_word \"pwho\" --output_dir ../models/")
    print("   python -m openwakeword.train --target_word \"peawho\" --output_dir ../models/")
    print("   python -m openwakeword.train --target_word \"pee hoo\" --output_dir ../models/")
    print("   python -m openwakeword.train --target_word \"pea who\" --output_dir ../models/")
    print("\nNote: If the above command fails, it is usually because 'piper-sample-generator'")
    print("is not compiled for your specific OS architecture (like Mac ARM64).")
    print("If you encounter errors generating synthetic audio locally, the most reliable")
    print("alternative is to use the official Google Colab notebook:")
    print("https://colab.research.google.com/github/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb")
    print("="*60)

if __name__ == "__main__":
    main()
