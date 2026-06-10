# Pihu Wake Word System

This repository contains the custom wake word detection system for **Pihu**, built using the incredibly fast, transfer-learning based [openWakeWord](https://github.com/dscripka/openWakeWord) library. 

This system allows your Mac (or any other device) to passively listen for the custom wake word "Pihu" (including various Indian-accented pronunciations) and trigger downstream actions.

## ⚙️ Prerequisites

- Python 3.10+
- [Conda](https://docs.conda.io/en/latest/) or `venv` for environment management.
- A working microphone.

## 🚀 Setup Instructions

Follow these steps to get the environment fully running on your local machine:

### 1. Create and Activate the Environment
It is highly recommended to use a Conda environment to avoid issues with audio dependencies like `PyAudio`.

```bash
# Create the environment
conda create -p ./venv_310 python=3.10

# Activate the environment
conda activate ./venv_310
```

### 2. Install Dependencies
Once inside your environment, install the necessary libraries:

```bash
pip install -r requirements.txt
pip install pyaudio
```

### 3. Download the Base Models
`openWakeWord` relies on two massive pre-trained Google speech models (`melspectrogram.onnx` and `embedding_model.onnx`) to process raw audio before feeding it to our custom `pihu.onnx` model.

Run the provided setup script to automatically download these base models into the correct directory:

```bash
python download_models.py
```

---

## 🎙️ Running the Detector

Once your environment is set up and your models are downloaded, you can start the microphone stream!

Ensure your custom trained model (`pihu.onnx`) is placed inside the `models/` directory, then run:

```bash
python detector.py
```

The script will:
1. Automatically load all `.onnx` models found in the `models/` folder.
2. Open a real-time microphone stream.
3. Print `WAKE WORD DETECTED: pihu` whenever you speak the wake word.

---

## 🧠 Training (Google Colab)

If you wish to re-train the Pihu model or train a new wake word entirely, it is recommended to do so in **Google Colab** to utilize their free T4 GPUs.

1. Clone this repository into your Colab notebook.
2. Ensure you bypass TensorFlow dependencies to avoid Python 3.12 compatibility crashes.
3. Generate clips, augment them, and train the model using `openwakeword/train.py`.
4. Export the resulting `.onnx` file and place it in the `models/` directory of your local project.
