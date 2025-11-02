# 🚗 ANPR / LPR - Gate Demo

Projeto educacional de Reconhecimento Automático de Placas (ANPR / LPR)
usando **Python (FastAPI)** + **OCR (EasyOCR)** + **Frontend HTML/JS**.

## 🧩 Tecnologias
- FastAPI
- EasyOCR
- OpenCV
- HTML/CSS/JS

## 🚀 Como rodar

### Backend
```bash
cd backend
uvicorn app:app --reload --port 8000
cd frontend
python -m http.server 8080
