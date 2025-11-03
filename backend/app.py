# ============================================
# ANPR / LPR - Gate Demo (Reconhecimento e Controle de Placas)
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
import sqlite3

# ============================================
# CONFIGURAÇÕES INICIAIS
# ============================================
os.makedirs("detected", exist_ok=True)
DB_PATH = "plates.db"

# Cria banco de dados se não existir
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT UNIQUE
        )
    """)
    conn.commit()
    conn.close()

init_db()

app = FastAPI()

# ============================================
# CONFIGURAÇÃO CORS
# ============================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# FUNÇÕES AUXILIARES
# ============================================
def save_plate(plate: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO plates (plate) VALUES (?)", (plate,))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def check_plate_exists(plate: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM plates WHERE plate = ?", (plate,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists


# ============================================
# ENDPOINT: UPLOAD DE IMAGEM + OCR
# ============================================
@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        plate_region = None
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            if 2 < aspect_ratio < 6 and 80 < w < 1000 and 25 < h < 300:
                y1 = max(0, y - 15)
                y2 = min(img_np.shape[0], y + h + 15)
                x1 = max(0, x - 25)
                x2 = min(img_np.shape[1], x + w + 25)
                plate_region = img_np[y1:y2, x1:x2]
                cv2.imwrite("detected/placa_crop.jpg", plate_region)
                print(f">>> Região da placa salva em detected/placa_crop.jpg (x={x1}, y={y1}, w={x2-x1}, h={y2-y1})")
                break

        if plate_region is None:
            plate_region = img_np
            print(">>> Nenhuma região típica de placa encontrada — usando imagem completa.")

        gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        gray_plate = cv2.equalizeHist(gray_plate)
        gray_plate = cv2.convertScaleAbs(gray_plate, alpha=1.7, beta=15)

        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(gray_plate, cv2.MORPH_CLOSE, kernel)

        thresh = cv2.adaptiveThreshold(
            morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 41, 15
        )

        cv2.imwrite("detected/placa_preprocessada.jpg", thresh)

        reader = easyocr.Reader(['en', 'pt'], gpu=False)
        result = reader.readtext(thresh)

        print(">>> Resultado bruto OCR:", result)

        texts = [r[1].strip().upper() for r in result]
        print(">>> Textos detectados:", texts)

        plate_pattern = re.compile(r'^[A-Z]{3}\d[A-Z0-9]\d{2}$')
        possible_plates = [t for t in texts if plate_pattern.match(t)]
        final_plate = possible_plates[0] if possible_plates else "N/A"

        if final_plate == "N/A":
            result2 = reader.readtext(plate_region)
            texts2 = [r[1].strip().upper() for r in result2]
            possible_plates2 = [t for t in texts2 if plate_pattern.match(t)]
            if possible_plates2:
                final_plate = possible_plates2[0]
            print(">>> Segunda tentativa OCR:", texts2)

        print(">>> Texto final interpretado:", final_plate)

        return JSONResponse({
            "status": "ok",
            "ocr_result": final_plate
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem: {e}")


# ============================================
# ENDPOINT: ADICIONAR PLACA MANUALMENTE
# ============================================
@app.post("/api/plates")
async def add_plate(plate: str):
    plate = plate.strip().upper()
    if not re.match(r'^[A-Z]{3}\d[A-Z0-9]\d{2}$', plate):
        raise HTTPException(status_code=400, detail="Formato de placa inválido.")
    success = save_plate(plate)
    if not success:
        return {"status": "duplicado", "message": "Placa já cadastrada."}
    return {"status": "ok", "message": f"Placa {plate} adicionada com sucesso."}


# ============================================
# ENDPOINT: CHECAR PLACA
# ============================================
@app.get("/api/check/{plate}")
async def check_plate(plate: str):
    plate = plate.strip().upper()
    found = check_plate_exists(plate)
    return {"status": "ok", "found": found}


# ============================================
# ENDPOINT: STATUS DA API
# ============================================
@app.get("/api/test")
async def test_api():
    return {"status": "API Online"}


# ============================================
# EXECUÇÃO LOCAL
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)