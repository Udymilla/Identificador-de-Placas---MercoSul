from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
from PIL import Image
import numpy as np
import traceback

app = FastAPI()

# Configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Banco de placas em memória
registered_plates = []

@app.get("/")
async def root():
    return {"status": "ok", "registered": registered_plates}

@app.get("/api/check/{plate}")
async def check_plate(plate: str):
    """Verifica se a placa está registrada"""
    if plate in registered_plates:
        return {"plate": plate, "status": "PERMITIDO"}
    return {"plate": plate, "status": "NEGADO"}

@app.post("/api/plates")
async def add_plate(plate: dict):
    """Adiciona uma nova placa à lista"""
    try:
        plate_number = plate.get("plate", "").upper().strip()
        if not plate_number:
            raise HTTPException(status_code=400, detail="Placa inválida")
        if plate_number not in registered_plates:
            registered_plates.append(plate_number)
        return {"added": plate_number, "registered": registered_plates}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=str(e))

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Recebe uma imagem e simula a leitura da placa via OCR"""
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        np_img = np.array(img)

        # Simulação de OCR (fixo para teste)
        recognized_plate = "ABC1D23"

        # Verifica se a placa é registrada
        status = "PERMITIDO" if recognized_plate in registered_plates else "NEGADO"

        return {"ocr": recognized_plate, "status": status}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    