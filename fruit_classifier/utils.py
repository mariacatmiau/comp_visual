import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from PIL import Image

IMG_SIZE_DEFAULT = 224

def load_model(model_path):
    try:
        return keras_load_model(model_path)
    except Exception as e:
        print("Erro ao carregar o modelo:", e)
        return None

def load_labels(labels_path):
    try:
        with open(labels_path, "r") as f:
            labels = json.load(f)
        if isinstance(labels, dict):
            labels = [labels[str(i)] for i in range(len(labels))]
        return labels
    except Exception as e:
        print("Erro ao carregar labels:", e)
        return None

def preprocess_image(image, img_size=IMG_SIZE_DEFAULT):
    img = image.convert("RGB").resize((img_size, img_size))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def topk_predictions(preds, class_names, k=3):
    probs = tf.nn.softmax(preds[0]).numpy()
    top_indices = probs.argsort()[-k:][::-1]
    return [(class_names[i], probs[i]) for i in top_indices]
