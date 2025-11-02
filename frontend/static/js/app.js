// ===========================
// ANPR Gate Demo Frontend
// ===========================

// Base da API — usa 127.0.0.1 (não localhost!)
const API_BASE = "http://127.0.0.1:8000/api";

// Referências de elementos da interface
const statusOverlay = document.getElementById("statusOverlay");
const events = document.getElementById("events");
const ocrResult = document.getElementById("ocrResult");
const ocrPreview = document.getElementById("ocrPreview");
const checkResult = document.getElementById("checkResult");
const plateInput = document.getElementById("plateInput");
const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const addBtn = document.getElementById("addBtn");
const newPlate = document.getElementById("newPlate");


// ===========================
// Funções auxiliares
// ===========================

function log(msg) {
  const ts = new Date().toLocaleTimeString();
  const line = `[${ts}] ${msg}`;
  console.log(line);
  events.innerText = line + "\n" + events.innerText;
}

function setStatus(text, color = "gray") {
  statusOverlay.innerText = text;
  statusOverlay.style.backgroundColor = color;
}

// ===========================
// Ações principais
// ===========================

// Checar placa manualmente
document.getElementById("checkBtn").onclick = async () => {
  const plate = plateInput.value.trim().toUpperCase();
  if (!plate) return alert("Digite uma placa no formato Mercosul (ex: ABC1D23)");

  try {
    const r = await fetch(API_BASE + "/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plate }),
    });

    if (!r.ok) throw new Error("Falha na comunicação com o backend");
    const data = await r.json();

    checkResult.innerText = data.allowed ? "LIBERADO" : "NEGADO";
    checkResult.style.color = data.allowed ? "lime" : "red";
    setStatus(data.allowed ? "LIBERADO" : "NEGADO", data.allowed ? "green" : "red");
    log(`Checagem: ${plate} → ${data.allowed ? "LIBERADO" : "NEGADO"}`);
  } catch (err) {
    log("Erro de conexão ao checar placa.");
    console.error(err);
  }
};

// Upload de imagem (OCR)
uploadBtn.onclick = async () => {
  const file = fileInput.files[0];
  if (!file) return alert("Escolha uma imagem antes de enviar!");

  const fd = new FormData();
  fd.append("file", file);

  try {
    const r = await fetch(API_BASE + "/upload", { method: "POST", body: fd });
    if (!r.ok) throw new Error("Falha no upload da imagem");

    const data = await r.json();
    ocrResult.innerText = `OCR: ${data.plate || "N/A"} → ${data.allowed ? "LIBERADO" : "NEGADO"}`;
    ocrResult.style.color = data.allowed ? "lime" : "red";

    if (data.image_url) {
      ocrPreview.src = data.image_url;
      ocrPreview.style.display = "block";
    }
    setStatus(data.allowed ? "LIBERADO" : "NEGADO", data.allowed ? "green" : "red");
    log(`OCR: ${data.plate || "N/A"} → ${data.allowed ? "LIBERADO" : "NEGADO"}`);
  } catch (err) {
    log("Erro no upload da imagem (500).");
    console.error(err);
  }
};

// Adicionar nova placa
addBtn.onclick = async () => {
  const plate = newPlate.value.trim().toUpperCase();
  if (!plate) return alert("Digite uma placa válida!");

  try {
    const r = await fetch(API_BASE + "/plates", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plate }),
    });

    if (!r.ok) throw new Error("Falha ao adicionar placa");
    const data = await r.json();

    const li = document.createElement("li");
    li.innerText = plate;
    plateList.appendChild(li);

    log(`Placa adicionada: ${plate}`);
  } catch (err) {
    log("Erro ao adicionar placa.");
    console.error(err);
  }
};
