import os
import pyaudio
import numpy as np
from openwakeword.model import Model
import time

# Directory containing your custom models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def get_custom_models():
    """Retrieve all .onnx model paths from the models directory."""
    if not os.path.exists(MODEL_DIR):
        print(f"Warning: Models directory '{MODEL_DIR}' not found. Please run train.py first.")
        return []
    
    models = []
    for file in os.listdir(MODEL_DIR):
        if file.endswith(".onnx"):
            models.append(os.path.join(MODEL_DIR, file))
    return models

def main():
    # 1. Load custom models
    custom_models = get_custom_models()
    
    if not custom_models:
        print("No custom .onnx models found in the 'models' directory.")
        print("Starting openwakeword with default models (alexa, hey mycroft) for testing...")
        # If no custom models, just load default ones for testing
        owwModel = Model(inference_framework="onnx")
    else:
        print(f"Loading {len(custom_models)} custom models: {[os.path.basename(m) for m in custom_models]}")
        # Initialize the openwakeword model with our custom paths
        owwModel = Model(wakeword_models=custom_models, inference_framework="onnx")

    # 2. Setup PyAudio microphone stream
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1280  # 1280 frames = 80ms of audio at 16000 Hz
    
    audio = pyaudio.PyAudio()
    
    print("\n--- Starting Microphone Stream ---")
    try:
        mic_stream = audio.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                frames_per_buffer=CHUNK)
    except IOError as e:
        print(f"Error opening microphone: {e}")
        print("Make sure your microphone is connected and accessible.")
        return

    print("Listening... (Press Ctrl+C to stop)")
    
    # 3. Main listening loop
    try:
        while True:
            # Read audio data from microphone
            audio_data = mic_stream.read(CHUNK, exception_on_overflow=False)
            
            # Convert raw bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Feed audio to openwakeword model
            # It expects audio to be 16kHz, 16-bit PCM
            prediction = owwModel.predict(audio_np)
            
            # Check predictions
            for mdl in owwModel.prediction_buffer.keys():
                # The prediction buffer stores the confidence scores.
                # If confidence > threshold, we consider it a detection.
                scores = list(owwModel.prediction_buffer[mdl])
                if scores and scores[-1] > 0.5: # 0.5 is a common threshold, adjust if needed
                    # To avoid continuous triggering, we clear the buffer or wait a bit
                    print(f"\n[{time.strftime('%H:%M:%S')}] WAKE WORD DETECTED: {mdl} (Score: {scores[-1]:.3f})")
                    # Clear buffer for this model to prevent re-triggering immediately
                    owwModel.prediction_buffer[mdl].clear()
                    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()
