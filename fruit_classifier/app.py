import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
# IMPORTANTE: Adicione esta linha
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# ========================================
# Função para carregar o modelo
# ========================================
def load_model_file():
    models_dir = "models"
    model = None

    for file_name in os.listdir(models_dir):
        if file_name.endswith(".keras") or file_name.endswith(".h5"):
            path = os.path.join(models_dir, file_name)
            try:
                # Tente carregar sem compile=False primeiro, pode ajudar
                model = tf.keras.models.load_model(path) 
                st.sidebar.success(f"✅ Modelo carregado com sucesso: {file_name}")
                return model
            except Exception as e:
                # Se falhar, tente com compile=False
                try:
                    model = tf.keras.models.load_model(path, compile=False)
                    st.sidebar.success(f"✅ Modelo carregado (compile=False): {file_name}")
                    return model
                except Exception as e2:
                    st.sidebar.warning(f"Falha ao carregar {file_name}: {e2}")

    st.sidebar.error("❌ Nenhum modelo válido encontrado. Verifique a pasta 'models/'.")
    return None

# ========================================
# Função para processar imagem (CORRIGIDA)
# ========================================
def preprocess_image(image, target_size=(224, 224)):
    # 1. Converte para RGB (caso seja PNG com transparência) e redimensiona
    image = image.convert("RGB").resize(target_size)
    # 2. Converte para array numpy
    image = np.array(image)
    # 3. Adiciona dimensão do batch (lote)
    image = np.expand_dims(image, axis=0)
    # 4. USA O PRÉ-PROCESSAMENTO DO MOBILENETV2 (Converte de [0, 255] para [-1, 1])
    image = mobilenet_preprocess(image)
    return image

# ========================================
# Carregar labels
# ========================================
def load_labels():
    try:
        with open("models/labels.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        st.sidebar.error("⚠️ Arquivo labels.json não encontrado em 'models/'.")
        return {}

# ========================================
# Interface Streamlit
# ========================================
st.set_page_config(page_title="Classificador de Frutas", page_icon="🍎", layout="centered")

st.title("🍎 Classificador de Frutas (MobileNetV2)")
st.write("Envie uma imagem de fruta e veja a previsão do modelo real do Colab!")

# Sidebar
st.sidebar.header("⚙️ Configurações")
# O tamanho da imagem DEVE ser 224, pois o modelo foi treinado com 224
img_size = 224
st.sidebar.info("Tamanho da imagem fixado em 224x224 (requerido pelo MobileNetV2).")
top_k = st.sidebar.slider("Top-K", 1, 5, 3)

# Carregar modelo e labels
model = load_model_file()
labels = load_labels()

uploaded_file = st.file_uploader("📸 Envie uma imagem (jpg/png):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Imagem recebida", use_column_width=True)

    if model and labels:
        with st.spinner("🔍 Classificando..."):
            # O tamanho da imagem é fixo, não usamos mais img_size da sidebar
            img_array = preprocess_image(image, (img_size, img_size))
            
            preds = model.predict(img_array)[0]
            top_indices = np.argsort(preds)[::-1][:top_k]

            st.success("✅ Classificação concluída!")
            st.subheader("📊 Resultados:")

            for i in top_indices:
                label = labels.get(str(i), f"Classe {i}")
                st.write(f"**{label}** — {preds[i]*100:.2f}%")
    else:
        st.error("❌ Nenhum modelo carregado. Coloque o arquivo `.h5` ou `.keras` na pasta 'models/'.")

else:
    st.info("📤 Envie uma imagem para começar.")
