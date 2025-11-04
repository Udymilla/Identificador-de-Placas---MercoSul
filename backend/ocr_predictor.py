# =========================================================
# OCR HÍBRIDO (EasyOCR + KNN Customizado)
# =========================================================
import cv2
import numpy as np
import easyocr
import os
import re

# ============================================
# CONFIGURAÇÃO GLOBAL
# ============================================
DATASET_DIR = os.path.join("backend", "ocr_chars")
MODEL_PATH = os.path.join("backend", "ocr_knn_model.npz")

# Cria diretório para armazenar dataset se ainda não existir
os.makedirs(DATASET_DIR, exist_ok=True)

# ============================================
# FUNÇÕES AUXILIARES DE TREINAMENTO
# ============================================
def extract_characters(image):
    """Recorta caracteres isolados de uma placa binarizada"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chars = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > 20 and w > 10:  # ignora ruído
            roi = thresh[y:y+h, x:x+w]
            roi = cv2.resize(roi, (20, 20))
            chars.append((x, roi))
    chars = sorted(chars, key=lambda c: c[0])
    return [roi for (_, roi) in chars]


def train_knn():
    """Treina um modelo KNN básico para OCR com base em amostras locais"""
    samples = []
    labels = []
    for label in os.listdir(DATASET_DIR):
        path = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(path):
            continue
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (20, 20)).reshape(-1).astype(np.float32)
            samples.append(img)
            labels.append(ord(label))  # usa o código ASCII do caractere
    if len(samples) == 0:
        print("⚠️ Nenhum caractere no dataset para treinar o KNN.")
        return None

    samples = np.array(samples)
    labels = np.array(labels)
    knn = cv2.ml.KNearest_create()
    knn.train(samples, cv2.ml.ROW_SAMPLE, labels)
    np.savez_compressed(MODEL_PATH, samples=samples, labels=labels)
    print(f"✅ Modelo KNN treinado com {len(samples)} amostras.")
    return knn


def load_knn():
    """Carrega o modelo KNN salvo ou cria um novo"""
    if os.path.exists(MODEL_PATH):
        data = np.load(MODEL_PATH)
        samples = data["samples"]
        labels = data["labels"]
        knn = cv2.ml.KNearest_create()
        knn.train(samples, cv2.ml.ROW_SAMPLE, labels)
        print("✅ Modelo KNN carregado.")
        return knn
    else:
        return train_knn()


# ============================================
# INICIALIZAÇÃO
# ============================================
knn_model = load_knn()
reader = easyocr.Reader(['en', 'pt'], gpu=False)

# ============================================
# FUNÇÃO PRINCIPAL DE OCR HÍBRIDO
# ============================================
def recognize_plate(image):
    """
    Tenta primeiro com EasyOCR.
    Se não detectar padrão de placa válido, tenta caractere a caractere com KNN.
    """
    try:
        # 1️⃣ EasyOCR
        result = reader.readtext(image)
        texts = [r[1].strip().upper() for r in result]
        print(">>> EasyOCR:", texts)
        plate_pattern = re.compile(r'^[A-Z]{3}\d[A-Z0-9]\d{2}$')
        for t in texts:
            t_clean = "".join(ch for ch in t if ch.isalnum())
            if plate_pattern.match(t_clean):
                print(">>> Placa reconhecida via EasyOCR:", t_clean)
                return t_clean

        # 2️⃣ Fallback com KNN
        if knn_model is None:
            print("⚠️ Modelo KNN não disponível.")
            return "N/A"

        chars = extract_characters(image)
        if not chars:
            print("⚠️ Nenhum caractere isolado detectado.")
            return "N/A"

        recognized = ""
        for ch in chars:
            sample = ch.reshape(-1).astype(np.float32)
            ret, result, neighbours, dist = knn_model.findNearest(sample.reshape(1, -1), k=3)
            recognized += chr(int(result[0][0]))

        recognized = recognized.upper()
        print(">>> KNN reconheceu:", recognized)
        return recognized if len(recognized) >= 6 else "N/A"

    except Exception as e:
        print("❌ Erro no OCR híbrido:", e)
        return "N/A"