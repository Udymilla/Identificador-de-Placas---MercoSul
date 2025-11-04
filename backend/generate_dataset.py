# ============================================
# Gerador de Dataset OCR (A–Z, 0–9) - KNN OCR
# ============================================
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Diretório de saída
base_dir = "backend/ocr_chars"
os.makedirs(base_dir, exist_ok=True)

# Caracteres de A–Z e 0–9
chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

# Fonte simples padrão
font = ImageFont.load_default()

for ch in chars:
    ch_dir = os.path.join(base_dir, ch)
    os.makedirs(ch_dir, exist_ok=True)
    for i in range(1, 21):  # 20 variações por caractere
        img = Image.new("L", (20, 20), color=255)
        draw = ImageDraw.Draw(img)
        # pequenas variações de posição e rotação
        angle = np.random.uniform(-10, 10)
        offset_x = np.random.randint(1, 5)
        offset_y = np.random.randint(1, 5)
        temp = Image.new("L", (20, 20), 255)
        d = ImageDraw.Draw(temp)
        d.text((offset_x, offset_y), ch, fill=0, font=font)
        rotated = temp.rotate(angle, expand=0, fillcolor=255)
        img.paste(rotated, (0, 0))
        arr = np.array(img)
        noise = np.random.randint(0, 40, arr.shape, dtype=np.uint8)
        arr = cv2.add(arr, noise)
        cv2.imwrite(os.path.join(ch_dir, f"{ch}_{i}.png"), arr)

print("✅ Dataset gerado com sucesso em backend/ocr_chars/")