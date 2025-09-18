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

# --- UI: CSS global (ponlo una sola vez) ---
st.markdown("""
<style>
.inline-row{display:flex; align-items:center; gap:.5rem; flex-wrap:wrap}
.spin{
  display:inline-block; width:1rem; height:1rem;
  border:2px solid rgba(0,0,0,.25); border-top-color: currentColor;
  border-radius:50%; animation:spin .8s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}
.tick{color:#16a34a; font-weight:700}
.muted{opacity:.7}
</style>
""", unsafe_allow_html=True)

st.title("Transcriptor de Vídeos con Whisper")
st.caption("Convierte vídeos de Google Drive o archivos subidos a transcripciones con marcas de tiempo y subtítulos SRT.")

# ----------------------- CONSTANTES -----------------------
BLOQUE_SEGUNDOS_DEFAULT = 4 * 60  # 4 minutos
EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4a", ".mp3", ".wav"}

# ----------------------- UTILIDADES -----------------------

# --- Helpers de UI ---
import time

def _fmt_duration(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h: return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"

def ui_start_processing(filename: str):
    """
    Crea/actualiza una línea de estado: 'Procesando: <file> ... [spinner]'
    Devuelve (placeholder, t0) para cerrar después.
    """
    ph = st.empty()
    ph.markdown(
        f"""<div class="inline-row">
              <div><b>Procesando</b>: {filename}…</div>
              <div class="spin"></div>
            </div>""",
        unsafe_allow_html=True
    )
    return ph, time.perf_counter()

def ui_end_processing(ph, filename: str, elapsed_sec: float):
    """
    Cambia el estado a '✅ Procesado' y muestra duración.
    """
    ph.markdown(
        f"""<div class="inline-row">
              <div><span class="tick">✅ Procesado</span> <b>{filename}</b></div>
              <div class="muted">({_fmt_duration(elapsed_sec)})</div>
            </div>""",
        unsafe_allow_html=True
    )

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
    ruta_txt = ruta_video.with_suffix(".txt")
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
    from googleapiclient.http import MediaIoBaseDownload  # <-- AÑADIR
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
    # Beam size en faster-whisper. (entre 1 y 10). Cuanto mayor más exactitud y más lento
    # CPU en Streamlit Cloud (sin GPU): empieza con 2–4.
    # Audio difícil / acentos / ruido: sube a 5–6.
    # Procesar muchos vídeos / ir más rápido: baja a 1–2.
    # Más de 7–8 rara vez compensa el coste.
    beam_size = st.number_input("Beam size", 1, 10, 5)
    vad_filter = st.checkbox("VAD filter", value=True)
    bloque_seg = st.number_input("Ventana de capítulo (segundos)", 60, 1800, BLOQUE_SEGUNDOS_DEFAULT, step=60)

    compute_type_cpu = st.selectbox("Compute type en CPU", ["int8", "int8_float16", "int16"], index=0)

    st.markdown("---")
    st.caption("Consejo: en Streamlit Cloud no hay GPU; usa **small** + **int8** para mejor rendimiento.")

# ----------------------- UI: SELECCIÓN FUENTE -----------------------
# archivos_locales = []   # [(Path, display_name)]
ss = st.session_state   #Añadido
ss.setdefault("archivos_locales", [])   #Añadido
ss.setdefault("uploader_key", 0)      # <- nuevo: para resetear el file_uploader
ss.setdefault("uploader_names", set())  # <- nuevo: nombres actualmente subidos por el widget
archivos_locales = ss["archivos_locales"]   # Añadido [(Path, display_name)]
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
    up = st.file_uploader(
        "Arrastra tus vídeos/audio",
        type=[e.lstrip(".") for e in EXTS],
        accept_multiple_files=True,
        key=f"uploader_{ss.uploader_key}"  # <- clave variable para poder resetear
    )

    # Nombres actualmente visibles en el widget (tras quitar con la “x”, ya no estarán aquí)
    curr_names = {f.name for f in (up or [])}

    # 1) Sincroniza eliminaciones hechas en el widget (las “x”)
    if ss.uploader_names:
        for i, (p, name) in enumerate(list(archivos_locales)):
            # solo limpiamos los que proceden del uploader
            if name in ss.uploader_names and name not in curr_names:
                archivos_locales.pop(i)
                ss.uploader_names.discard(name)

    # 2) Añade SOLO los nuevos (evita duplicados en cada rerun)
    new_files = [f for f in (up or []) if f.name not in ss.uploader_names]
    for f in new_files:
        dest = carpeta_trabajo / f.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():  # evita reescribir en cada rerun
            with open(dest, "wb") as fh:
                fh.write(f.read())
        archivos_locales.append((dest, f.name))
    ss.uploader_names.update(f.name for f in new_files)


# Añadido
# ---- Acciones sobre la lista de archivos preparada ----
# ---- Acciones sobre la lista de archivos preparada ----
list_box = st.container()
with list_box:
    if archivos_locales:
        st.success(f"Archivos preparados: {len(archivos_locales)}")
        st.markdown("")  # separador para evitar solape con el expander

        with st.expander("Ver lista de archivos preparados", expanded=False):
            for i, (p, name) in enumerate(list(archivos_locales)):
                c1, c2 = st.columns([9, 1])
                with c1:
                    st.markdown(f"• **{name}**")
                with c2:
                    if st.button("🗑️", key=f"del_{i}", help=f"Eliminar {name}"):
                        archivos_locales.pop(i)
                        ss.uploader_names.discard(name)
                        st.rerun()
    else:
        st.info("No hay archivos preparados todavía.")

col_a, col_b = st.columns(2)
with col_a:
    start_now = st.button("▶️ Iniciar transcripción", type="primary", disabled=(len(archivos_locales) == 0))
with col_b:
    if st.button("🧹 Vaciar lista"):
        archivos_locales.clear()
        ss.uploader_names.clear()
        ss.uploader_key += 1   # <- reinicia el widget file_uploader
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            st.rerun()


#Añadido


# ----------------------- TRANSCRIPCIÓN -----------------------
# DESPUÉS
# if archivos_locales:
if 'start_now' in locals() and start_now:
    # --- 2.1 Depurar la lista en session_state (quitar no existentes y duplicados) ---
    depurada = []
    vistos = set()
    for p, name in archivos_locales:
        p = Path(p)
        key = (str(p.resolve()), name)
        if p.exists() and key not in vistos:
            vistos.add(key)
            depurada.append((p, name))
    # Actualizamos IN-PLACE para que el contador también refleje la depuración
    archivos_locales.clear()
    archivos_locales.extend(depurada)

    # --- 2.2 Congelar el snapshot a procesar (evita que entren borrados recientes) ---
    files_to_process = archivos_locales[:]  # copia superficial

    if not files_to_process:
        st.warning("No hay archivos válidos en la lista.")
    else:
        model = load_model(model_name, compute_type_cpu)
        resumen_rows = []
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for local_path, display_name in files_to_process:
                st.write("---")
                #st.write(f"**Procesando:** {display_name}")
                ph, t0 = ui_start_processing(display_name)
                # ⬇️ A partir de aquí, mantén tu lógica tal cual (transcribir, guardar SRT/TXT, añadir al ZIP, etc.)
                try:
                    segments_list, info, palabras_total = transcribir_archivo(
                        local_path, model, force_lang.strip() or None, beam_size, vad_filter
                    )

                    # Guardar SRT
                    srt_path = out_dir / (Path(display_name).stem + ".srt")
                    srt_path = guardar_srt(segments_list, srt_path)

                    # Bloques 4 min
                    bloques = agrupar_por_bloques(segments_list, bloque_s=bloque_seg)
                    txt_path = out_dir / (Path(display_name).stem + ".txt")
                    txt_path = guardar_transcripcion_4min(txt_path, Path(display_name).stem, bloques)
                    elapsed = time.perf_counter() - t0
                    ui_end_processing(ph, display_name, elapsed)
                    # Duración aprox por último segmento
                    if segments_list:
                        dur = max(float(s.end) for s in segments_list if s.end is not None)
                    else:
                        dur = 0.0

                    st.success(
                        f"✓ Listo: {display_name} | Duración: {human_time(dur)} | Idioma detectado: {getattr(info,'language', 'auto')} | Tiempo de Procesado: {_fmt_duration(elapsed)}"
                    )

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
                        "bloques_min": len(bloques),
                        "bloques_con_texto": sum(1 for b in bloques if b["texto"]),
                        "txt_min": txt_path.name,
                        "srt": srt_path.name,
                        "tiempo_proceso_seg": round(elapsed, 2),
                        "tiempo_proceso_hhmmss": _fmt_duration(elapsed),
                    })
                except Exception as e:
                    st.error(f"✗ Error procesando {display_name}: {e}")
                    continue

    # Resumen CSV
    #df = pd.DataFrame(resumen_rows)
    #csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    st.subheader("Descargas")
    st.download_button("📥 Descargar ZIP (.srt + .txt)", data=zip_buffer.getvalue(), file_name="transcripciones.zip")
    #st.download_button("📊 Descargar resumen_transcripciones.csv", data=csv_bytes, file_name="resumen_transcripciones.csv", mime="text/csv")
else:
    st.info("Selecciona una fuente y prepara al menos un archivo para transcribir.")
