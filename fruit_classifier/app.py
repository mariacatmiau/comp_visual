import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
import io

# ========================================
# ‼️ MUDANÇA CRÍTICA AQUI ‼️
# Importar a função de pré-processamento EXATA do MobileNetV2
# ========================================
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# ========================================
# Função para carregar o modelo
# ========================================
@st.cache_resource
def load_model_file():
    """Carrega o modelo .h5 da pasta /models e o armazena em cache."""
    models_dir = "models"
    model = None
    
    # Procurar por .h5 ou .keras
    h5_files = [f for f in os.listdir(models_dir) if f.endswith(".h5")]
    keras_files = [f for f in os.listdir(models_dir) if f.endswith(".keras")]
    
    # Dar preferência ao .h5 que sabemos que funciona
    model_files = h5_files + keras_files 

    if not model_files:
        st.sidebar.error("❌ Nenhum ficheiro .h5 ou .keras encontrado na pasta 'models/'.")
        return None

    model_path = os.path.join(models_dir, model_files[0])
    
    try:
        # Carregar o modelo compilado é desnecessário para inferência
        model = tf.keras.models.load_model(model_path, compile=False)
        st.sidebar.success(f"✅ Modelo carregado: {model_files[0]}")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Falha ao carregar {model_files[0]}. Erro: {e}")
        return None

# ========================================
# Função para processar imagem (CORRIGIDA)
# ========================================
def preprocess_image(image, target_size=(224, 224)):
    """
    Prepara a imagem para o MobileNetV2.
    1. Redimensiona para (224, 224)
    2. Converte para array numpy
    3. Adiciona dimensão de batch
    4. Aplica a normalização de [-1, 1] do MobileNetV2
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    image = image.resize(target_size)
    img_array = np.array(image)
    
    # Adicionar a dimensão do "batch" (lote)
    img_expanded = np.expand_dims(img_array, axis=0)
    
    # Aplicar a normalização do MobileNetV2 (pixels de -1 a 1)
    processed_img = mobilenet_preprocess(img_expanded)
    
    return processed_img

# ========================================
# Carregar labels
# ========================================
@st.cache_data
def load_labels():
    """Carrega o labels.json e o armazena em cache."""
    labels_path = os.path.join("models", "labels.json")
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_dict = json.load(f)
            # Converte o dicionário {"0": "Fruta"} para uma lista ["Fruta", ...]
            labels_list = [labels_dict[str(i)] for i in range(len(labels_dict))]
            st.sidebar.info(f"✅ Labels carregados ({len(labels_list)} classes).")
            return labels_list
    except Exception as e:
        st.sidebar.error(f"⚠️ labels.json não encontrado ou inválido. Erro: {e}")
        return []

# ========================================
# Interface Streamlit
# ========================================
st.set_page_config(page_title="Classificador de Frutas", page_icon="🍎", layout="centered")

st.title("🍎 Classificador de Frutas (MobileNetV2)")
st.write(f"Treinado com {len(load_labels())} classes de frutas comuns no Brasil.")

# Sidebar
st.sidebar.header("⚙️ Configurações")
top_k = st.sidebar.slider("Mostrar Top-K previsões:", 1, 5, 3)

# Carregar modelo e labels
model = load_model_file()
class_names = load_labels() # Agora é uma LISTA

uploaded_file = st.file_uploader("📸 Envie uma imagem (jpg/png):", type=["jpg", "jpeg", "png"])
camera_img = st.camera_input("Ou tire uma foto 📷")

# Determinar qual imagem usar
img_bytes = None
if uploaded_file:
    img_bytes = uploaded_file.getvalue()
elif camera_img:
    img_bytes = camera_img.getvalue()


if img_bytes is not None:
    image = Image.open(io.BytesIO(img_bytes))
    st.image(image, caption="🖼️ Imagem recebida", use_column_width=True)

    if model and class_names:
        with st.spinner("🔍 Classificando..."):
            # Usar a nova função de pré-processamento
            img_array = preprocess_image(image, (224, 224))
            
            preds = model.predict(img_array)[0]
            
            # Pegar os índices das Top-K previsões
            top_indices = np.argsort(preds)[::-1][:top_k]

            st.success("✅ Classificação concluída!")
            st.subheader(f"📊 Resultados (Top {top_k}):")

            for i in top_indices:
                label = class_names[i] # Acessar pela lista
                prob = preds[i]
                st.write(f"**{label}**: {prob:.2%}")
    else:
        st.error("❌ Modelo ou labels não carregados. Verifique a pasta 'models/'.")
else:
    st.info("📤 Envie uma imagem para começar.")

