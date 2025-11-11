# ============================================
# ANPR / LPR - Gate Demo (Reconhecimento e Controle de Placas)
# ============================================

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.ocr_predictor import recognize_plate
import easyocr
import numpy as np
import traceback
import io
import cv2
from PIL import Image
import re
import os
import sqlite3

# --- PASTAS E ARQUIVOS AUXILIARES ---
os.makedirs("detected", exist_ok=True)
CASCADE_PATH = os.path.join("backend", "models", "haarcascade_russian_plate_number.xml")

# --- EASYOCR: carregar só uma vez (evita lentidão a cada requisição) ---
reader = easyocr.Reader(['en', 'pt'], gpu=False)

# --- Normalização simples para padrão Mercosul (ABC1D23) ---
MERCOSUL_RE = re.compile(r'^[A-Z]{3}\d[A-Z0-9]\d{2}$')

def normalize_plate(txt: str) -> str:
    if not txt:
        return ""
    t = "".join(ch for ch in txt.upper() if ch.isalnum())
    if len(t) < 7:
        return t
    # Correções comuns do OCR nas posições numéricas
    # ABC 1 D 23
    # 012 3 4 56
    chars = list(t[:7])
    want = ["A","A","A","N","A","N","N"]  # A=letra, N=número
    swapL = {"0":"O","1":"I","5":"S","8":"B","2":"Z","6":"G"}
    swapN = {"O":"0","I":"1","L":"1","S":"5","B":"8","Z":"2","G":"6"}
    for i,ch in enumerate(chars):
        if want[i] == "A" and not ch.isalpha():
            chars[i] = swapL.get(ch, ch)
        if want[i] == "N" and not ch.isdigit():
            chars[i] = swapN.get(ch, ch)
    fixed = "".join(chars)
    return fixed

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


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        # ========= CARREGAR E PRÉ-PROCESSAR =========
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
                y1, y2 = max(0, y - 15), min(img_np.shape[0], y + h + 15)
                x1, x2 = max(0, x - 25), min(img_np.shape[1], x + w + 25)
                plate_region = img_np[y1:y2, x1:x2]
                cv2.imwrite("detected/placa_crop.jpg", plate_region)
                print(f">>> Região da placa salva em detected/placa_crop.jpg (x={x1}, y={y1}, w={x2-x1}, h={y2-y1})")
                break

        if plate_region is None:
            plate_region = img_np
            print(">>> Nenhuma região típica de placa encontrada — usando imagem completa.")

        # ========= PRÉ-PROCESSAMENTO PARA OCR =========
        gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        gray_plate = cv2.equalizeHist(gray_plate)
        gray_plate = cv2.convertScaleAbs(gray_plate, alpha=1.7, beta=15)
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(gray_plate, cv2.MORPH_CLOSE, kernel)

        # Testa modo invertido se necessário
        thresh = cv2.adaptiveThreshold(
            morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 41, 15
        )
        cv2.imwrite("detected/debug_entrada_ocr.jpg", thresh)

        # ========= EASY OCR =========
        reader = easyocr.Reader(['en', 'pt'], gpu=False)
        result = reader.readtext(thresh)
        easy_texts = [r[1].strip().upper() for r in result]
        print(">>> EasyOCR:", easy_texts)

        # ========= FALLBACK PARA KNN =========
        final_plate = "N/A"
        if easy_texts:
            final_plate = easy_texts[0]
        else:
            try:
                from ocr_predictor import ocr_knn_predict
                knn_result = ocr_knn_predict(plate_region)
                print(">>> KNN reconheceu:", knn_result)
                if knn_result:
                    final_plate = knn_result
            except Exception as e:
                print(f"[WARN] Falha ao usar KNN: {e}")

        # ========= CORREÇÃO DE ERROS COMUNS =========
        substitutions = {
            'O': '0', 'Q': '0', 'D': '0',
            'I': '1', 'L': '1',
            'Z': '2',
            'S': '5',
            'B': '8'
        }
        corrected = "".join(substitutions.get(ch, ch) for ch in final_plate)
        if corrected != final_plate:
            print(f">>> Placa corrigida de {final_plate} para {corrected}")
            final_plate = corrected

        # ========= TREINAMENTO CONTÍNUO (APRENDIZADO) =========
        if final_plate != "N/A":
            try:
                from ocr_predictor import update_knn_model
                update_knn_model(plate_region, final_plate)
                print(f">>> Modelo KNN atualizado com a placa {final_plate}")
            except Exception as e:
                print(f"[WARN] Não foi possível atualizar KNN: {e}")

        # ========= CHECAGEM NO BANCO =========
        found = check_plate_exists(final_plate)
        status = "AUTORIZADO" if found else "NEGADO"
        if final_plate == "N/A":
            status = "NEGADO"

        print(f">>> Resultado final: {final_plate} → {status}")

        return JSONResponse({
            "status": "ok",
            "ocr_result": final_plate,
            "authorized": status
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem: {e}")
# ============================================
# ENDPOINT: ADICIONAR PLACA MANUALMENTE
# ============================================
from fastapi import Body

@app.post("/api/plates")
async def add_plate(plate: str = None, payload: dict = Body(None)):
    # aceita ?plate=ABC1D23 ou body {"plate":"ABC1D23"}
    if plate is None and payload and "plate" in payload:
        plate = payload["plate"]

    if not plate:
        raise HTTPException(status_code=400, detail="Informe a placa.")

    plate = normalize_plate(plate)
    if not MERCOSUL_RE.match(plate):
        raise HTTPException(status_code=400, detail="Formato de placa inválido (padrão Mercosul: ABC1D23).")

    success = save_plate(plate)
    if not success:
        return {"status": "duplicado", "message": "Placa já cadastrada.", "plate": plate}
    return {"status": "ok", "message": f"Placa {plate} adicionada com sucesso.", "plate": plate}
# ============================================
# ENDPOINT: CHECAR PLACA
# ============================================
@app.get("/api/check/{plate}")
async def check_plate(plate: str):
    plate = plate.strip().upper()
    found = check_plate_exists(plate)
    result = "AUTORIZADO" if found else "NEGADO"
    print(f">>> Verificação manual: {plate} → {result}")
    return {
        "status": "ok",
        "plate": plate,
        "found": found,
        "result": result
    }
# ============================================
# ENDPOINT: STATUS DA API
# ============================================
@app.get("/api/test")
async def test_api():
    return {"status": "API Online"}

@app.get("/api/plates/list")
async def list_plates():
    # lista simples do banco
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