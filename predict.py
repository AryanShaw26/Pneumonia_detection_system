import cv2
import numpy as np
import tensorflow as tf
import sys
import os
import json

IMG_SIZE = 128
LABELS = ['NORMAL', 'PNEUMONIA']
MODEL_PATH = os.path.join("model_weight", "vgg_unfrozen.h5")

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Read in color
    if img is None:
        raise ValueError(f"Could not read the image at {image_path}")
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_array = img_rgb.reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
    return img_array

def predict_image(image_path):
    preprocessed_img = preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)[0]
    print(f"[predict.py] Raw model predictions: {predictions} (type: {type(predictions)}, shape: {getattr(predictions, 'shape', None)})", file=sys.stderr)
    # Handle binary (sigmoid) or categorical (softmax) outputs
    if len(predictions.shape) == 0 or predictions.shape == () or (hasattr(predictions, 'shape') and predictions.shape == (1,)):
        # Single sigmoid output
        prob = float(predictions)
        print(f"[predict.py] Interpreted as sigmoid. Prob: {prob}", file=sys.stderr)
        if prob >= 0.7:
            predicted_label = 'PNEUMONIA'
            confidence = prob * 100
        else:
            predicted_label = 'NORMAL'
            confidence = (1 - prob) * 100
    elif len(predictions) == 2:
        # Softmax output
        print(f"[predict.py] Interpreted as softmax. Probabilities: {predictions}", file=sys.stderr)
        predicted_index = int(np.argmax(predictions))
        predicted_label = LABELS[predicted_index]
        confidence = float(predictions[predicted_index]) * 100
        print(f"[predict.py] Argmax index: {predicted_index}, Label: {predicted_label}, Confidence: {confidence}", file=sys.stderr)
        # Apply threshold for pneumonia
        if predicted_label == 'PNEUMONIA' and confidence < 70:
            print(f"[predict.py] Confidence for PNEUMONIA below threshold (70%). Reporting as NORMAL.", file=sys.stderr)
            predicted_label = 'NORMAL'
            confidence = 100 - confidence
    else:
        raise ValueError(f"Unexpected model output shape: {predictions.shape}")
    return predicted_label, confidence

if __name__ == "__main__":
    print("[predict.py] Script started", file=sys.stderr)
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>", file=sys.stderr)
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"[predict.py] Image path: {image_path}", file=sys.stderr)
    try:
        label, conf = predict_image(image_path)
        print(json.dumps({"label": label, "confidence": conf}))
    except Exception as e:
        print(f"[predict.py] Exception: {e}", file=sys.stderr)
        print(json.dumps({"error": str(e)}))