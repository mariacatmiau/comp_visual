import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
import io

# Importar a função de pré-processamento correta

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# ========================================
# CONSTANTES
# ========================================
# Estes são os nomes dos ficheiros do seu treino V7 (o de 29 classes)
MODEL_NAME = "model_real.h5"
LABELS_NAME = "labels.json"
MODELS_DIR = "models"

# ========================================
# Função para carregar o modelo
# ========================================
@st.cache_resource
def load_model_file():
    """Carrega o modelo .h5 treinado (V7)."""
    model_path = os.path.join(MODELS_DIR, MODEL_NAME)
    
    if not os.path.exists(model_path):
        st.sidebar.error(f"❌ Modelo '{MODEL_NAME}' não encontrado na pasta 'models/'.")
        st.sidebar.info("Certifique-se de que copiou o 'model_real.h5' e 'labels.json' (de 29 classes) do Colab para a pasta 'models/'.")
        return None
        
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.sidebar.success(f"✅ Modelo carregado: {MODEL_NAME}")
        return model
    except Exception as e:
        st.sidebar.error(f"Falha ao carregar {MODEL_NAME}: {e}")
        return None

# ========================================
# Função para carregar labels
# ========================================
@st.cache_data
def load_labels_file():
    """Carrega os labels .json (V7)."""
    labels_path = os.path.join(MODELS_DIR, LABELS_NAME)
    
    if not os.path.exists(labels_path):
        st.sidebar.error(f"❌ Labels '{LABELS_NAME}' não encontrados na pasta 'models/'.")
        return None
        
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            # O ficheiro V7 é um DICIONÁRIO (ex: {"0": "Maçã", ...})
            labels_dict = json.load(f)
        
        # Converter para lista para facilitar o uso ( ["Maçã", "Banana", ...] )
        labels_list = [labels_dict[str(i)] for i in range(len(labels_dict))]
        
        st.sidebar.success(f"✅ Labels carregados ({len(labels_list)} classes).")
        return labels_list
    except Exception as e:
        st.sidebar.error(f"Falha ao carregar {LABELS_NAME}: {e}")
        return None

# ========================================
# Função para processar imagem
# ========================================
def preprocess_image_for_model(image_pil, target_size=(224, 224)):
    """Prepara a imagem PIL para o modelo MobileNetV2 (normaliza para [-1, 1])."""
    img_resized = image_pil.resize(target_size)
    img_array = np.array(img_resized)
    
    # Se a imagem for PNG com canal alfa (transparência), remover
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    img_expanded = np.expand_dims(img_array, axis=0)
    
    # Aplicar a normalização do MobileNetV2 (converte de [0, 255] para [-1, 1])
    img_processed = mobilenet_preprocess(img_expanded)
    
    return img_processed

# ========================================
# Interface Streamlit
# ========================================
st.set_page_config(page_title="Classificador de Frutas", page_icon="🍎", layout="centered")

st.title("🍎 Classificador de Frutas (V7)")
st.write("Envie uma imagem de fruta para classificação!")

# Sidebar
st.sidebar.header("⚙️ Sobre")
st.sidebar.info("Este modelo (MobileNetV2) foi treinado no dataset Fruits-360.")

# Carregar modelo e labels
model = load_model_file()
class_names = load_labels_file() # class_names agora é uma LISTA

# Input do utilizador (APENAS O FILE UPLOADER)
uploaded_file = st.file_uploader("📸 Envie uma imagem (jpg/png):", type=["jpg", "jpeg", "png"])

# Se uma imagem foi recebida
if uploaded_file: # Simplificado
    try:
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        
        # ========================================
        # MUDANÇA AQUI (Correção do Aviso)
        # ========================================
        # O comando 'use_column_width=True' foi substituído
        # pelo novo comando 'use_container_width=True'
        st.image(image, caption="🖼️ Imagem recebida", use_container_width=True)
        # ========================================
        # FIM DA MUDANÇA
        # ========================================

        if model and class_names:
            with st.spinner("🔍 Classificando..."):
                # 1. Processar a imagem
                img_array = preprocess_image_for_model(image)
                
                # 2. Fazer a previsão
                preds = model.predict(img_array)[0]
                
                # 3. Obter os top 3
                top_k = 3
                top_indices = np.argsort(preds)[::-1][:top_k]

                st.success("✅ Classificação concluída!")
                st.subheader("📊 Resultados:")

                # 4. Mostrar os resultados
                for i in top_indices:
                    label = class_names[i]
                    prob = preds[i]
                    st.write(f"**{label.capitalize()}** — {prob*100:.2f}%")
        else:
            st.error("❌ Modelo ou labels não carregados. Verifique os erros na sidebar.")

    except Exception as e:
        st.error(f"Erro ao processar a imagem: {e}")
else:
    st.info("📤 Envie uma imagem para começar.")

