# ============================================
# ANPR / LPR - Gate Demo (Reconhecimento e Controle de Placas)
# ============================================

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import os
import io
import re
import cv2
import sqlite3
import traceback
import numpy as np
from PIL import Image

# --- Tentar carregar YOLO (opcional) ---
YOLO_AVAILABLE = False
yolo_model = None
try:
    from ultralytics import YOLO
    model_path = os.path.join("backend", "models", "yolo_plate.pt")  # ajuste o nome se o seu .pt for outro
    if os.path.exists(model_path):
        yolo_model = YOLO(model_path)
        YOLO_AVAILABLE = True
        print(f"[YOLO] Modelo carregado de {model_path}")
    else:
        print(f"[YOLO] Modelo não encontrado em {model_path}, usando fallback com OpenCV.")
except Exception as e:
    print(f"[YOLO] Não foi possível carregar YOLO: {e}")
    YOLO_AVAILABLE = False

# --- Tentar carregar EasyOCR (opcional) ---
EASYOCR_AVAILABLE = False
reader = None
try:
    import easyocr
    reader = easyocr.Reader(["en", "pt"], gpu=False)
    EASYOCR_AVAILABLE = True
    print("[EasyOCR] Reader carregado.")
except Exception as e:
    print(f"[EasyOCR] Não foi possível carregar EasyOCR: {e}")
    EASYOCR_AVAILABLE = False

# --------------------------------------------
# PASTAS E BANCO
# --------------------------------------------
os.makedirs("detected", exist_ok=True)
DB_PATH = "plates.db"

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

# --------------------------------------------
# NORMALIZAÇÃO DE PLACA (MERCOSUL)
# --------------------------------------------
MERCOSUL_RE = re.compile(r"^[A-Z]{3}\d[A-Z0-9]\d{2}$")

def normalize_plate(txt: str) -> str:
    """Normaliza texto para formato próximo de ABC1D23."""
    if not txt:
        return ""
    t = "".join(ch for ch in txt.upper() if ch.isalnum())
    if len(t) < 7:
        return t

    # queremos A A A 1 A 2 3
    # posições 0,1,2,4 letras; 3,5,6 números
    chars = list(t[:7])
    want = ["A", "A", "A", "N", "A", "N", "N"]  # A=letra, N=número

    swapL = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}
    swapN = {"O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"}

    for i, ch in enumerate(chars):
        if want[i] == "A" and not ch.isalpha():
            chars[i] = swapL.get(ch, ch)
        if want[i] == "N" and not ch.isdigit():
            chars[i] = swapN.get(ch, ch)

    fixed = "".join(chars)
    return fixed

# --------------------------------------------
# FUNÇÕES DE BANCO
# --------------------------------------------
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

# --------------------------------------------
# DETECÇÃO DA REGIÃO DA PLACA (YOLO ou fallback OpenCV)
# --------------------------------------------
def detect_plate_region(img_np: np.ndarray) -> np.ndarray:
    """
    Tenta detectar a placa com YOLO.
    Se não conseguir ou YOLO não estiver disponível, usa fallback simples com contornos.
    Retorna um recorte (crop) da placa (ou a imagem inteira se não achar nada).
    """
    h, w = img_np.shape[:2]

    # 1) YOLO
    if YOLO_AVAILABLE and yolo_model is not None:
        try:
            # YOLO espera RGB
            results = yolo_model(img_np[:, :, ::-1], verbose=False)
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                # Pega a box com maior confiança
                best_idx = int(boxes.conf.argmax())
                xyxy = boxes.xyxy[best_idx].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = img_np[y1:y2, x1:x2]
                cv2.imwrite("detected/placa_crop.jpg", crop)
                print(f">>> [YOLO] Placa detectada em detected/placa_crop.jpg (x={x1}, y={y1}, w={x2-x1}, h={y2-y1})")
                return crop
        except Exception as e:
            print(f"[YOLO] Erro ao detectar placa: {e}")

    # 2) Fallback com Canny + contornos
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        aspect_ratio = ww / float(hh)
        if 2 < aspect_ratio < 6 and 80 < ww < 1000 and 25 < hh < 300:
            y1, y2 = max(0, y - 15), min(h, y + hh + 15)
            x1, x2 = max(0, x - 25), min(w, x + ww + 25)
            crop = img_np[y1:y2, x1:x2]
            cv2.imwrite("detected/placa_crop.jpg", crop)
            print(f">>> [CANNY] Placa detectada em detected/placa_crop.jpg (x={x1}, y={y1}, w={x2-x1}, h={y2-y1})")
            return crop

    print(">>> Nenhuma região típica de placa encontrada — usando imagem completa.")
    return img_np

# --------------------------------------------
# OCR DA PLACA
# --------------------------------------------
def ocr_plate(plate_img: np.ndarray) -> str:
    """
    Roda OCR (EasyOCR se disponível) + normalização de placa.
    """
    if not EASYOCR_AVAILABLE or reader is None:
        print("[OCR] EasyOCR não disponível.")
        return "N/A"

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.convertScaleAbs(gray, alpha=1.7, beta=15)

    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    thresh = cv2.adaptiveThreshold(
        morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 41, 15
    )
    cv2.imwrite("detected/placa_preprocessada.jpg", thresh)

    result = reader.readtext(thresh)
    texts = [r[1].strip().upper() for r in result]
    print(">>> Textos detectados OCR:", texts)

    if not texts:
        return "N/A"

    # pega o maior texto (normalmente a placa)
    best = max(texts, key=len)
    best_norm = normalize_plate(best)
    print(f">>> Texto bruto: {best} | Normalizado: {best_norm}")

    if MERCOSUL_RE.match(best_norm):
        return best_norm
    return best_norm or "N/A"

# --------------------------------------------
# FASTAPI
# --------------------------------------------

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# ENDPOINT: UPLOAD DE IMAGEM (OCR + YOLO)
# ============================================
@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)[:, :, ::-1]  # PIL RGB -> OpenCV BGR

        # 1) Detectar placa
        plate_region = detect_plate_region(img_np)

        # 2) OCR
        plate_text = ocr_plate(plate_region)

        # 3) Checar no banco
        allowed = False
        if plate_text != "N/A":
            allowed = check_plate_exists(plate_text)

        status_str = "LIBERADO" if allowed else "NEGADO"
        print(f">>> Resultado final upload: {plate_text} → {status_str}")

        # opcionalmente poderia expor uma URL de imagem se você servir /detected pelo backend
        image_url = None

        return JSONResponse({
            "status": "ok",
            "plate": plate_text,
            "allowed": allowed,
            "image_url": image_url
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem: {e}")

# ============================================
# ENDPOINT: ADICIONAR PLACA (JSON {plate})
# ============================================
@app.post("/api/plates")
async def add_plate(payload: dict = Body(...)):
    plate = payload.get("plate", "")
    plate = normalize_plate(plate)

    if not plate:
        raise HTTPException(status_code=400, detail="Informe a placa.")
    if not MERCOSUL_RE.match(plate):
        raise HTTPException(status_code=400, detail="Formato de placa inválido (padrão Mercosul: ABC1D23).")

    success = save_plate(plate)
    if not success:
        return {"status": "duplicado", "message": "Placa já cadastrada.", "plate": plate}
    return {"status": "ok", "message": f"Placa {plate} adicionada com sucesso.", "plate": plate}

# ============================================
# ENDPOINT: CHECAR PLACA (POST /api/check com JSON {plate})
# ============================================
@app.post("/api/check")
async def check_plate_body(payload: dict = Body(...)):
    plate = payload.get("plate", "")
    plate = normalize_plate(plate)

    if not plate:
        raise HTTPException(status_code=400, detail="Informe a placa.")

    found = check_plate_exists(plate)
    result = "LIBERADO" if found else "NEGADO"
    print(f">>> Verificação manual: {plate} → {result}")

    return {
        "status": "ok",
        "plate": plate,
        "allowed": found,
        "result": result,
    }

# (opcional) manter também a rota antiga GET /api/check/{plate}
@app.get("/api/check/{plate}")
async def check_plate_path(plate: str):
    plate = normalize_plate(plate)
    found = check_plate_exists(plate)
    result = "LIBERADO" if found else "NEGADO"
    print(f">>> Verificação manual (GET): {plate} → {result}")
    return {
        "status": "ok",
        "plate": plate,
        "allowed": found,
        "result": result,
    }

# ============================================
# ENDPOINTS EXTRAS
# ============================================
@app.get("/api/test")
async def test_api():
    return {"status": "API Online"}

@app.get("/api/plates/list")
async def list_plates():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT plate FROM plates ORDER BY plate")
    plates = [row[0] for row in cur.fetchall()]
    conn.close()
    return {"status": "ok", "plates": plates}

@app.delete("/api/plates/delete/{plate}")
async def delete_plate(plate: str):
    plate = normalize_plate(plate)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM plates WHERE plate = ?", (plate,))
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    return {"status": "ok", "deleted": deleted, "plate": plate}

# ============================================
# EXECUÇÃO LOCAL
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)