import tensorflow as tf
from tensorflow.keras import layers, models
import json, os

os.makedirs("models", exist_ok=True)

# Cria CNN simples compatível com o app
model = models.Sequential([
    layers.Input(shape=(224,224,3)),
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(9, activation='softmax')
])

model.save("models/model.h5")

labels = {
    "0": "abacaxi",
    "1": "banana",
    "2": "laranja",
    "3": "maçã",
    "4": "morango",
    "5": "melancia",
    "6": "uva",
    "7": "limão",
    "8": "kiwi"
}

with open("models/labels.json", "w") as f:
    json.dump(labels, f)

print("✅ Modelo fictício criado em models/model.h5")
