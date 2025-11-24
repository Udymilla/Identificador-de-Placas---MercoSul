# ============================================
# ANPR / LPR - Gate Demo (YOLOv5 + EasyOCR)
# ============================================

import os
import io
import re
import sqlite3
import traceback

import cv2
import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ==== MODELOS EXTERNOS ====
# YOLOv5 via ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Não foi possível importar ultralytics/YOLO: {e}")
    YOLO_AVAILABLE = False

# EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Não foi possível importar easyocr: {e}")
    EASYOCR_AVAILABLE = False

# ============================================
# CONFIGURAÇÕES E CAMINHOS
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTED_DIR = os.path.join(BASE_DIR, "..", "detected")
os.makedirs(DETECTED_DIR, exist_ok=True)

DB_PATH = os.path.join(BASE_DIR, "..", "plates.db")
YOLO_WEIGHTS = os.path.join(BASE_DIR, "models", "plate_yolo.pt")

# Regex padrão Mercosul (ABC1D23)
MERCOSUL_RE = re.compile(r'^[A-Z]{3}\d[A-Z0-9]\d{2}$')

# ============================================
# FUNÇÕES DE PLACA / BANCO
# ============================================

def normalize_plate(txt: str) -> str:
    """Normaliza texto para padrão de placa Mercosul (ABC1D23) o melhor possível."""
    if not txt:
        return ""
    # remove tudo que não é letra ou número
    t = "".join(ch for ch in txt.upper() if ch.isalnum())
    if len(t) < 7:
        return t

    # queremos 7 chars -> ABC1D23
    t = t[:7]
    chars = list(t)
    # A = letra, N = número (padrão Mercosul)
    want = ["A", "A", "A", "N", "A", "N", "N"]

    # trocas comuns (letra->número e número->letra)
    swapL = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}
    swapN = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
             "Z": "2", "S": "5", "B": "8", "G": "6"}

    for i, ch in enumerate(chars):
        if want[i] == "A" and not ch.isalpha():
            chars[i] = swapL.get(ch, ch)
        if want[i] == "N" and not ch.isdigit():
            chars[i] = swapN.get(ch, ch)

    fixed = "".join(chars)
    return fixed


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


def save_plate(plate: str) -> bool:
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
# CARREGAR MODELOS (YOLO + EasyOCR)
# ============================================

yolo_model = None
if YOLO_AVAILABLE and os.path.exists(YOLO_WEIGHTS):
    try:
        yolo_model = YOLO(YOLO_WEIGHTS)
        print(f"[YOLO] Modelo carregado de {YOLO_WEIGHTS}")
    except Exception as e:
        print(f"[WARN] Falha ao carregar YOLO: {e}")
        yolo_model = None
else:
    print("[YOLO] Modelo YOLO não disponível (verifique ultralytics e plate_yolo.pt).")

easy_reader = None
if EASYOCR_AVAILABLE:
    try:
        easy_reader = easyocr.Reader(['en', 'pt'], gpu=False)
        print("[EasyOCR] Reader carregado.")
    except Exception as e:
        print(f"[WARN] Falha ao inicializar EasyOCR: {e}")
        easy_reader = None
else:
    print("[EasyOCR] EasyOCR não disponível.")


# ============================================
# INICIALIZA FASTAPI
# ============================================

init_db()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# FUNÇÃO DE OCR A PARTIR DE UMA REGIÃO (NUMPY)
# ============================================

def ocr_from_region(plate_img: np.ndarray) -> str | None:
    """Executa pré-processamento + EasyOCR em uma região de placa."""
    if easy_reader is None:
        return None

    # Salva para debug
    cv2.imwrite(os.path.join(DETECTED_DIR, "plate_crop.jpg"), plate_img)

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.convertScaleAbs(gray, alpha=1.6, beta=10)

    # filtro morfológico
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # limiar adaptativo
    thresh = cv2.adaptiveThreshold(
        morph, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41, 15
    )

    cv2.imwrite(os.path.join(DETECTED_DIR, "plate_preprocessed.jpg"), thresh)

    # OCR
    results = easy_reader.readtext(thresh, detail=0)
    texts = [t.strip().upper() for t in results if t.strip()]
    print("[OCR] Textos detectados:", texts)

    if not texts:
        return None

    # tenta achar algo que pareça placa Mercosul
    candidates = []
    for t in texts:
        norm = normalize_plate(t)
        if MERCOSUL_RE.match(norm):
            candidates.append(norm)

    if candidates:
        return candidates[0]

    # se nada bateu exato, tenta a primeira leitura normalizada
    norm0 = normalize_plate(texts[0])
    return norm0 if norm0 else None


# ============================================
# ENDPOINT: UPLOAD DE IMAGEM (YOLO + OCR)
# ============================================

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        # lê bytes e converte para OpenCV
        img_bytes = await file.read()
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # salva imagem original de debug
        cv2.imwrite(os.path.join(DETECTED_DIR, "entrada_original.jpg"), img_np)

        plate_region = None

        # ========== 1) DETECÇÃO COM YOLO ==========
        if yolo_model is not None:
            try:
                results = yolo_model(img_np, imgsz=640, verbose=False)
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    # pega box com maior confiança
                    best = boxes[boxes.conf.argmax()]
                    x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_np.shape[1], x2), min(img_np.shape[0], y2)
                    plate_region = img_np[y1:y2, x1:x2]
                    print(f"[YOLO] Placa detectada em x={x1}, y={y1}, w={x2-x1}, h={y2-y1}")
                else:
                    print("[YOLO] Nenhuma placa detectada.")
            except Exception as e:
                print(f"[WARN] Erro ao executar YOLO: {e}")

        # ========== 2) FALLBACK: CANNY/CONTORNOS ==========
        if plate_region is None:
            print("[CANNY] Usando fallback de contornos.")
            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 100, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            best_box = None
            best_area = 0
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                aspect = w / float(h)
                area = w * h
                if 2.0 < aspect < 6.0 and area > best_area and 60 < w < 1000 and 25 < h < 300:
                    best_area = area
                    best_box = (x, y, w, h)
            if best_box:
                x, y, w, h = best_box
                x1 = max(0, x - 15)
                y1 = max(0, y - 15)
                x2 = min(img_np.shape[1], x + w + 15)
                y2 = min(img_np.shape[0], y + h + 15)
                plate_region = img_np[y1:y2, x1:x2]
                print(f"[CANNY] Placa candidata: x={x1}, y={y1}, w={x2-x1}, h={y2-y1}")
            else:
                print("[CANNY] Nenhuma região de placa encontrada.")

        plate_text = None
        allowed = False

        if plate_region is not None:
            # ==== OCR da região ====
            plate_text = ocr_from_region(plate_region)
            print(f"[UPLOAD] Texto lido (normalizado): {plate_text}")

            if plate_text and MERCOSUL_RE.match(plate_text):
                allowed = check_plate_exists(plate_text)

        # resposta para o frontend
        return JSONResponse({
            "plate": plate_text if plate_text else None,
            "allowed": allowed
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem: {e}")


# ============================================
# ENDPOINT: CHECAGEM MANUAL (/api/check)
# ============================================

@app.post("/api/check")
async def api_check(payload: dict = Body(...)):
    plate_raw = (payload.get("plate") or "").strip().upper()
    if not plate_raw:
        raise HTTPException(status_code=400, detail="Informe a placa.")

    plate = normalize_plate(plate_raw)
    if not MERCOSUL_RE.match(plate):
        # ainda assim fazemos a checagem, mas avisamos
        print(f"[CHECK] Placa em formato estranho: {plate_raw} -> {plate}")

    found = check_plate_exists(plate)
    print(f"[CHECK] Verificação manual: {plate} -> {'LIBERADO' if found else 'NEGADO'}")

    return {
        "plate": plate,
        "allowed": bool(found)
    }


# ============================================
# ENDPOINT: ADICIONAR PLACA (/api/plates)
# ============================================

@app.post("/api/plates")
async def add_plate(payload: dict = Body(...)):
    plate_raw = (payload.get("plate") or "").strip().upper()
    if not plate_raw:
        raise HTTPException(status_code=400, detail="Informe a placa.")

    plate = normalize_plate(plate_raw)
    if not MERCOSUL_RE.match(plate):
        raise HTTPException(
            status_code=400,
            detail="Formato de placa inválido (use padrão Mercosul, ex: ABC1D23)."
        )

    success = save_plate(plate)
    if not success:
        return {"status": "duplicado", "message": "Placa já cadastrada.", "plate": plate}

    print(f"[PLATE] Cadastrada nova placa: {plate}")
    return {"status": "ok", "message": f"Placa {plate} adicionada com sucesso.", "plate": plate}


# ============================================
# ENDPOINTS AUXILIARES (LISTAR / TESTE)
# ============================================

@app.get("/api/plates/list")
async def list_plates():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT plate FROM plates ORDER BY plate")
    plates = [row[0] for row in cur.fetchall()]
    conn.close()
    return {"status": "ok", "plates": plates}


@app.get("/api/test")
async def test_api():
    return {"status": "API Online"}


# ============================================
# EXECUÇÃO LOCAL DIRETA (opcional)
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)