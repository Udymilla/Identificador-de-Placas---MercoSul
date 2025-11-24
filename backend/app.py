# ============================================
# ANPR / LPR - Gate Demo (PaddleOCR + FastAPI)
# ============================================

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sqlite3
import numpy as np
import cv2
import io
import os
import re
from paddleocr import PaddleOCR
import traceback

# --------------------------------------------
# CONFIGURAÇÃO DE PASTAS
# --------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTED_DIR = os.path.join(BASE_DIR, "..", "detected")
os.makedirs(DETECTED_DIR, exist_ok=True)

DB_PATH = os.path.join(BASE_DIR, "..", "plates.db")

# --------------------------------------------
# PADRÃO DE PLACA MERCOSUL
# ABC1D23
# --------------------------------------------
MERCOSUL_RE = re.compile(r'^[A-Z]{3}\d[A-Z0-9]\d{2}$')

def normalize_plate(text):
    """Normaliza texto para formato Mercosul (ABC1D23)"""
    if not text:
        return ""

    text = "".join([c for c in text.upper() if c.isalnum()])
    if len(text) != 7:
        return text

    # A,A,A,N,A,N,N
    want = ["A","A","A","N","A","N","N"]

    swapL = {"0": "O", "1": "I", "2": "Z", "5":"S", "6":"G", "8":"B"}
    swapN = {"O":"0", "I":"1", "L":"1", "Z":"2", "S":"5", "G":"6", "B":"8"}

    chars = list(text)
    for i,ch in enumerate(chars):
        if want[i]=="A" and not ch.isalpha():
            chars[i] = swapL.get(ch, ch)
        if want[i]=="N" and not ch.isdigit():
            chars[i] = swapN.get(ch, ch)

    return "".join(chars)

# --------------------------------------------
# BANCO DE DADOS
# --------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS plates(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate TEXT UNIQUE
    )
    """)
    conn.commit()
    conn.close()

def save_plate(plate):
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

def check_plate_exists(plate):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM plates WHERE plate = ?", (plate,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists

init_db()

# --------------------------------------------
# INICIALIZA PADDLE OCR
# --------------------------------------------
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en"  # funciona bem para padrão Mercosul
)

print("[OCR] PaddleOCR carregado com sucesso.")


# --------------------------------------------
# FASTAPI
# --------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --------------------------------------------
# ENDPOINT: UPLOAD /api/upload
# --------------------------------------------
@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        cv2.imwrite(os.path.join(DETECTED_DIR, "upload_raw.jpg"), img)

        # ---------------------------------
        # 1) RODAR OCR COMPLETO NA IMAGEM
        # ---------------------------------
        result = ocr.ocr(img, cls=True)

        texts = []
        if result:
            for line in result:
                if line:
                    for box in line:
                        text = box[1][0]
                        texts.append(text.upper())

        print("[OCR] Textos brutos:", texts)

        # ---------------------------------
        # 2) NORMALIZAR E TENTAR ACHAR PLACA
        # ---------------------------------
        candidates = []
        for t in texts:
            n = normalize_plate(t)
            if MERCOSUL_RE.match(n):
                candidates.append(n)

        final_plate = candidates[0] if candidates else None

        print("[OCR] Placa interpretada:", final_plate)

        # ---------------------------------
        # 3) VERIFICAR NO BANCO
        # ---------------------------------
        allowed = False
        if final_plate:
            allowed = check_plate_exists(final_plate)

        return {
            "plate": final_plate,
            "allowed": allowed
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------
# /api/check (manual)
# --------------------------------------------
@app.post("/api/check")
async def check_manual(payload: dict = Body(...)):
    raw = payload.get("plate","").strip().upper()
    plate = normalize_plate(raw)

    if not MERCOSUL_RE.match(plate):
        return {"plate": plate, "allowed": False}

    allowed = check_plate_exists(plate)
    return {"plate": plate, "allowed": allowed}


# --------------------------------------------
# /api/plates (adicionar)
# --------------------------------------------
@app.post("/api/plates")
async def add_plate_api(payload: dict = Body(...)):
    raw = payload.get("plate","").strip().upper()
    plate = normalize_plate(raw)

    if not MERCOSUL_RE.match(plate):
        raise HTTPException(status_code=400, detail="Formato inválido. Use ABC1D23")

    if save_plate(plate):
        return {"status": "ok", "plate": plate}
    else:
        return {"status": "duplicado", "plate": plate}


# --------------------------------------------
# /api/test
# --------------------------------------------
@app.get("/api/test")
async def test():
    return {"status": "API Online"}