# app.py
# Streamlit app para transcribir vídeos (bloques de 4 min) con faster-whisper
# - Fuente: Google Drive (Service Account) o subida de archivos
# - Salida: .srt + *_transcript_4min.txt + resumen_transcripciones.csv

import io
import os
import math
import zipfile
from pathlib import Path
from datetime import timedelta

import streamlit as st
import pandas as pd

# ==== Google Drive (Service Account) ====
from typing import List, Dict, Optional
#from google.oauth2 import service_account
#from googleapiclient.discovery import build
#from googleapiclient.http import MediaIoBaseDownload

# ==== Whisper ====
from faster_whisper import WhisperModel

# ----------------------- CONFIG UI -----------------------
st.set_page_config(page_title="Transcriptor de Vídeos (4 min)", page_icon="🎧", layout="wide")
st.title("Transcriptor de Vídeos con Whisper (bloques de 4 minutos)")
st.caption("Convierte vídeos de Google Drive o archivos subidos a transcripciones con marcas cada 4 minutos y subtítulos SRT.")

# ----------------------- CONSTANTES -----------------------
BLOQUE_SEGUNDOS_DEFAULT = 4 * 60  # 4 minutos
EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4a", ".mp3", ".wav"}

# ----------------------- UTILIDADES -----------------------
def human_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return str(timedelta(seconds=seconds))

def agrupar_por_bloques(segments, bloque_s=240):
    """
    segments: lista de objetos faster-whisper con .start, .end, .text
    Devuelve lista de dicts: {inicio, fin, texto}
    """
    bloques = []
    segs = list(segments) if segments else []
    if not segs:
        return bloques
    total_fin = max(int(math.ceil(s.end)) for s in segs if s.end is not None)
    num_bloques = max(1, math.ceil(total_fin / bloque_s))
    for i in range(num_bloques):
        ini, fin = i * bloque_s, min((i + 1) * bloque_s, total_fin)
        textos = []
        for s in segs:
            if s.start < fin and s.end > ini:
                textos.append((s.text or "").strip())
        bloques.append({"inicio": ini, "fin": fin, "texto": " ".join(t for t in textos if t).strip()})
    return bloques

def srt_time(t: float) -> str:
    ms = int(round((t - int(t)) * 1000))
    t_int = int(t)
    # HH:MM:SS,mmm con cero relleno
    return str(timedelta(seconds=t_int)).rjust(8, "0") + f",{ms:03d}"

def guardar_srt(segments, ruta_destino: Path) -> Path:
    ruta_srt = ruta_destino.with_suffix(".srt")
    with open(ruta_srt, "w", encoding="utf-8") as f:
        for idx, s in enumerate(segments, start=1):
            start = max(0.0, float(s.start or 0))
            end   = max(start, float(s.end or (start + 0.01)))
            f.write(f"{idx}\n{srt_time(start)} --> {srt_time(end)}\n{(s.text or '').strip()}\n\n")
    return ruta_srt

def guardar_transcripcion_4min(ruta_video: Path, titulo: str, bloques) -> Path:
    ruta_txt = ruta_video.with_name(f"{ruta_video.stem}_transcript_4min.txt")
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write(f"Título: {titulo}\n\n")
        for b in bloques:
            f.write(f"[{human_time(b['inicio'])} - {human_time(b['fin'])}]\n")
            f.write((b["texto"] if b["texto"] else "(sin contenido reconocido)"))
            f.write("\n\n")
    return ruta_txt

# ------------------ CARGA DEL MODELO (cacheado) ------------------
@st.cache_resource(show_spinner=False)
def load_model(model_name: str, compute_type_cpu: str):
    # Detecta dispositivo y ajusta tipo de cómputo
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass

    ct = "float16" if device == "cuda" else compute_type_cpu  # en CPU: int8 por defecto
    st.info(f"Modelo: {model_name} | Dispositivo: {device} | compute_type: {ct}")
    return WhisperModel(model_name, device=device, compute_type=ct)

def transcribir_archivo(local_path: Path, model, force_lang: Optional[str], beam_size: int, vad_filter: bool):
    """
    Devuelve: segments(list), info(obj), palabras_total(int)
    """
    kwargs = dict(beam_size=beam_size, vad_filter=vad_filter)
    if force_lang:
        segments, info = model.transcribe(str(local_path), language=force_lang, **kwargs)
    else:
        segments, info = model.transcribe(str(local_path), **kwargs)
    segments_list = list(segments)
    palabras_total = sum(len((s.text or "").strip().split()) for s in segments_list if s.text)
    return segments_list, info, palabras_total

# ---------------- Google Drive helpers (Service Account) ----------------

def get_drive_service():
    """
    Requiere en secrets:
    [gcp_service_account] -> JSON de la cuenta de servicio
    """
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ModuleNotFoundError:
        st.error(
            "Faltan dependencias de Google. Añade a requirements.txt: "
            "`google-auth`, `google-api-python-client`, `google-auth-httplib2` y vuelve a desplegar."
        )
        st.stop()

    if "gcp_service_account" not in st.secrets:
        st.error("Falta el bloque [gcp_service_account] en secrets. Ve a Settings → Secrets y pégalo.")
        st.stop()

    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scopes
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def listar_archivos_drive(drive, folder_id: str) -> List[Dict]:
    """Lista vídeos/audio en una carpeta por folder_id."""
    q = f"'{folder_id}' in parents and trashed=false and (mimeType contains 'video/' or mimeType contains 'audio/')"
    results = []
    page_token = None
    while True:
        resp = drive.files().list(
            q=q,
            fields="nextPageToken, files(id, name, mimeType, size, createdTime, modifiedTime)",
            pageToken=page_token,
            pageSize=1000,
            orderBy="name_natural"
        ).execute()
        results.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return results

def descargar_archivo_drive(drive, file_id: str, destino: Path) -> Path:
    request = drive.files().get_media(fileId=file_id)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        pbar = st.progress(0, text=f"Descargando {destino.name} ...")
        i = 0
        while not done:
            status, done = downloader.next_chunk()
            i += 1
            if status:
                pbar.progress(min(1.0, status.progress()))
        pbar.empty()
    return destino

# ----------------------- UI: SIDEBAR -----------------------
with st.sidebar:
    st.header("Parámetros")

    fuente = st.radio("Fuente de archivos", ["Google Drive (Service Account)", "Subir archivos"], index=0)

    model_name = st.selectbox("Modelo Whisper (faster-whisper)", ["small", "base", "medium", "large-v3"], index=0)
    force_lang = st.text_input("Forzar idioma (ej. es) [vacío = autodetección]", value="")
    beam_size = st.number_input("Beam size", 1, 10, 5)
    vad_filter = st.checkbox("VAD filter", value=True)
    bloque_seg = st.number_input("Ventana de capítulo (segundos)", 60, 1800, BLOQUE_SEGUNDOS_DEFAULT, step=60)

    compute_type_cpu = st.selectbox("Compute type en CPU", ["int8", "int8_float16", "int16"], index=0)

    st.markdown("---")
    st.caption("Consejo: en Streamlit Cloud no hay GPU; usa **small** + **int8** para mejor rendimiento.")

# ----------------------- UI: SELECCIÓN FUENTE -----------------------
archivos_locales = []   # [(Path, display_name)]
carpeta_trabajo = Path("work")
out_dir = Path("output")
out_dir.mkdir(exist_ok=True)

if fuente.startswith("Google Drive"):
    drive = get_drive_service()
    folder_id = st.text_input("Google Drive Folder ID", placeholder="p.ej. 1AbCDeFgHiJkLmNoPq...", value="")
    st.caption("Comparte la carpeta con el email de la **Service Account** de tus secrets. Abre la carpeta en Drive y copia el ID de la URL.")
    if folder_id:
        with st.spinner("Listando archivos en la carpeta..."):
            files = listar_archivos_drive(drive, folder_id)
        if not files:
            st.info("No se han encontrado vídeos/audio en la carpeta indicada.")
        else:
            df = pd.DataFrame([
                {
                    "name": f["name"],
                    "mimeType": f["mimeType"],
                    "size_MB": round(int(f.get("size", 0)) / (1024 * 1024), 2) if f.get("size") else None,
                    "id": f["id"],
                } for f in files
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.success(f"Archivos encontrados: {len(df)}")

            to_process = st.multiselect(
                "Selecciona archivos a procesar",
                options=[f["id"] for f in files],
                format_func=lambda fid: next((f["name"] for f in files if f["id"] == fid), fid),
            )

            if to_process and st.button("⬇️ Descargar selección"):
                for fid in to_process:
                    name = next(f["name"] for f in files if f["id"] == fid)
                    local_path = carpeta_trabajo / name
                    descargar_archivo_drive(drive, fid, local_path)
                    archivos_locales.append((local_path, name))

elif fuente == "Subir archivos":
    up = st.file_uploader("Arrastra tus vídeos/audio", type=[e.lstrip(".") for e in EXTS], accept_multiple_files=True)
    for f in up or []:
        dest = carpeta_trabajo / f.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(f.read())
        archivos_locales.append((dest, f.name))

# ----------------------- TRANSCRIPCIÓN -----------------------
if archivos_locales:
    model = load_model(model_name, compute_type_cpu)

    resumen_rows = []
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for local_path, display_name in archivos_locales:
            st.write("---")
            st.write(f"**Procesando:** {display_name}")
            try:
                segments_list, info, palabras_total = transcribir_archivo(
                    local_path, model, force_lang.strip() or None, beam_size, vad_filter
                )

                # Guardar SRT
                srt_path = out_dir / (Path(display_name).stem + ".srt")
                srt_path = guardar_srt(segments_list, srt_path)

                # Bloques 4 min
                bloques = agrupar_por_bloques(segments_list, bloque_s=bloque_seg)
                txt_path = out_dir / (Path(display_name).stem + "_transcript_4min.txt")
                txt_path = guardar_transcripcion_4min(txt_path, Path(display_name).stem, bloques)

                # Duración aprox por último segmento
                if segments_list:
                    dur = max(float(s.end) for s in segments_list if s.end is not None)
                else:
                    dur = 0.0

                st.success(f"✓ Listo: {display_name} | Duración aprox: {human_time(dur)} | Idioma detectado: {getattr(info,'language', 'auto')}")

                # Añadir a ZIP
                zf.write(srt_path, arcname=srt_path.name)
                zf.write(txt_path, arcname=txt_path.name)

                resumen_rows.append({
                    "archivo_video": display_name,
                    "duracion_seg": round(dur, 2),
                    "duracion_hhmmss": human_time(dur),
                    "idioma": getattr(info, "language", None),
                    "prob_idioma": round(getattr(info, "language_probability", 0.0), 4) if getattr(info, "language_probability", None) else None,
                    "palabras": palabras_total,
                    "bloques_4min": len(bloques),
                    "bloques_con_texto": sum(1 for b in bloques if b["texto"]),
                    "txt_4min": txt_path.name,
                    "srt": srt_path.name
                })

            except Exception as e:
                st.error(f"✗ Error procesando {display_name}: {e}")

    # Resumen CSV
    df = pd.DataFrame(resumen_rows)
    csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    st.subheader("Descargas")
    st.download_button("📥 Descargar ZIP (.srt + _transcript_4min.txt)", data=zip_buffer.getvalue(), file_name="transcripciones.zip")
    st.download_button("📊 Descargar resumen_transcripciones.csv", data=csv_bytes, file_name="resumen_transcripciones.csv", mime="text/csv")
else:
    st.info("Selecciona una fuente y prepara al menos un archivo para transcribir.")
