import urllib.request
import os

out_dir = "/Users/mayankjha/Documents/Projects/PIHU/Pihu_Tests/wakeup/openWakeWord_repo/openwakeword/resources/models"
os.makedirs(out_dir, exist_ok=True)

models = {
    "embedding_model.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
    "melspectrogram.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx"
}

for name, url in models.items():
    print(f"Downloading {name} from {url}...")
    urllib.request.urlretrieve(url, os.path.join(out_dir, name))
print("Downloads completed successfully!")
