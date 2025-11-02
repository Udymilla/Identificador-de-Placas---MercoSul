# ============================================
# ANPR / LPR - Gate Demo (Reconhecimento de Placas)
# ============================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import easyocr
import numpy as np
import traceback
import io
import cv2
from PIL import Image
import re
import os

# Cria pasta "detected" se não existir
os.makedirs("detected", exist_ok=True)

app = FastAPI()

# ================================
# Configuração do CORS
# ================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Endpoint principal: upload e OCR da imagem
# ============================================
@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        # === Leitura da imagem ===
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        # === Pré-processamento ===
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 200)

        # === Detecção de contornos (região da placa) ===
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        plate_region = None
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            if 2 < aspect_ratio < 6 and 100 < w < 1000 and 30 < h < 300:
                # Região provável de placa
                # Adiciona margem de segurança para capturar melhor a placa
                y1 = max(0, y - 10)
                y2 = min(img_np.shape[0], y + h + 10)
                x1 = max(0, x - 20)
                x2 = min(img_np.shape[1], x + w + 20)
                plate_region = img_np[y1:y2, x1:x2]

                cv2.imwrite("detected/placa_crop.jpg", plate_region)
                print(f">>> Região da placa salva em detected/placa_crop.jpg (x={x1}, y={y1}, w={x2-x1}, h={y2-y1})")
                # Salvar recorte para visualização
                cv2.imwrite("detected/placa_crop.jpg", plate_region)
                print(f">>> Região da placa salva em detected/placa_crop.jpg (x={x}, y={y}, w={w}, h={h})")
                break

        if plate_region is None:
            plate_region = img_np
            print(">>> Nenhuma região típica de placa encontrada — usando imagem completa.")

        # === Conversão para escala de cinza + limiar adaptativo ===
        gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        gray_plate = cv2.equalizeHist(gray_plate)
        gray_plate = cv2.convertScaleAbs(gray_plate, alpha=1.5, beta=10)
        thresh = cv2.adaptiveThreshold(
            gray_plate, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            41, 15
        )

        # === OCR (Reconhecimento de caracteres) ===
        reader = easyocr.Reader(['en', 'pt'], gpu=False)
        result = reader.readtext(thresh)

        print(">>> Resultado bruto OCR:", result)

        texts = [r[1].strip().upper() for r in result]
        print(">>> Textos detectados:", texts)

        # === Filtrar somente padrões de placas Mercosul ===
        plate_pattern = re.compile(r'^[A-Z]{3}\d[A-Z0-9]\d{2}$')
        possible_plates = [t for t in texts if plate_pattern.match(t)]
        final_plate = possible_plates[0] if possible_plates else "N/A"

        print(">>> Texto final interpretado:", final_plate)

        # === Retorno da API ===
        return JSONResponse({
            "status": "ok",
            "ocr_result": final_plate
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem: {e}")

# ============================================
# Inicialização do servidor
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)