#!/usr/bin/env python3
"""
SkyAR Demo Server - Local Version
A FastAPI-based web interface for the SkyAR video sky replacement system
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Optional

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# =============================
# ✅ FastAPI app configuration
# =============================

app = FastAPI(
    title="SkyAR Demo",
    description="Dynamic Sky Replacement in Videos",
    version="1.0",
    # Taille maximale des requêtes : 10 Mo
    max_request_size=310 * 1024 * 1024
)

# =============================
# ✅ CORS (autorisations web)
# =============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Tu peux restreindre à ton domaine si tu veux
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# ✅ Vérification dossier templates/static
# =============================

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Crée automatiquement les dossiers utiles s’ils n’existent pas
for d in ["uploads", "outputs", "temp"]:
    os.makedirs(BASE_DIR / d, exist_ok=True)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for uploads and outputs (local paths)
# === Chemins absolus ===
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_ZIPS_DIR = BASE_DIR / "temp_zips"
STATUS_FILE = BASE_DIR / "skyar_processing_status.pkl"

# Crée les dossiers si absents
for d in [UPLOAD_DIR, OUTPUT_DIR, TEMP_ZIPS_DIR]:
    d.mkdir(exist_ok=True)

# Mount static files and templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# Processing status storage (use file-based storage to persist across restarts)
import pickle
STATUS_FILE = Path("./skyar_processing_status.pkl")

def load_processing_status():
    """Load processing status from file"""
    try:
        if STATUS_FILE.exists():
            with open(STATUS_FILE, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Error loading status: {e}")
    return {}

def save_processing_status():
    """Save processing status to file"""
    try:
        with open(STATUS_FILE, 'wb') as f:
            pickle.dump(processing_status, f)
    except Exception as e:
        print(f"Error saving status: {e}")

processing_status = load_processing_status()

# Available sky templates - Natural blue skies only
SKY_TEMPLATES = {
    "bluesky1": {
        "name": "Blue Sky with Clouds",
        "description": "Natural blue sky with white clouds",
        "file": "bluesky1.jpg"
    },
    "bluesky2": {
        "name": "Clear Blue Sky",
        "description": "Clear blue sky with wispy clouds",
        "file": "bluesky2.jpg"
    },
    "bluesky3": {
        "name": "Cloudy Blue Sky",
        "description": "Blue sky with scattered clouds",
        "file": "bluesky3.jpg"
    },
    "bluesky4": {
        "name": "Serene Blue Sky",
        "description": "Peaceful blue sky gradient",
        "file": "bluesky4.jpg"
    }
}


class ProcessingRequest(BaseModel):
    video_id: str
    sky_template: str
    auto_light_matching: bool = False
    relighting_factor: float = 0.8
    recoloring_factor: float = 0.5
    halo_effect: bool = True


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main demo page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "sky_templates": SKY_TEMPLATES
    })


@app.post("/api/test-upload-size")
async def test_upload_size(file: UploadFile = File(...)):
    """Test endpoint to determine proxy upload limits"""
    try:
        content = await file.read()
        size = len(content)
        
        return {
            "success": True,
            "filename": file.filename,
            "size": size,
            "size_mb": round(size / (1024 * 1024), 2),
            "message": f"Upload successful - {file.filename} ({round(size / (1024 * 1024), 2)} MB)"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

from PIL import Image
import io

@app.post("/api/upload-single")
async def upload_single_file(file: UploadFile = File(...)):
    try:
        if not file.filename:
            return {"success": False, "error": "No file provided"}

        ext = Path(file.filename).suffix.lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
        allowed = image_extensions + video_extensions

        if ext not in allowed:
            return {"success": False, "error": "Unsupported file type"}

        # ✅ Lire tout le fichier (mobile compatible)
        content = await file.read()
        total_size = len(content)

        max_size = 11 * 1024 * 1024  # 10 MB limite

        if total_size == 0:
            return {"success": False, "error": "Empty file"}
        if total_size > max_size:
            return {"success": False, "error": "File too large (max 10MB)"}

        file_type = "image" if ext in image_extensions else "video"
        video_id = str(uuid.uuid4())
        upload_path = UPLOAD_DIR / f"{video_id}{ext}"

        if file_type == "image":
            image = Image.open(io.BytesIO(content))

            # ✅ Convertir HEIC / iPhone images → RGB automatiquement
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")

            # ✅ Redimensionnement intelligent (garde qualité)
            max_dim = 1920
            w, h = image.size
            if max(w, h) > max_dim:
                scale = max_dim / float(max(w, h))
                image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            # ✅ Sauvegarde compressée (optimisée mobile)
            image.save(upload_path, format="JPEG", quality=92)

        else:
            # ✅ Vidéo → sauvegarde brute
            async with aiofiles.open(upload_path, 'wb') as f:
                await f.write(content)

        processing_status[video_id] = {
            "status": "uploaded",
            "filename": file.filename,
            "file_path": str(upload_path),
            "file_type": file_type,
            "progress": 0,
            "message": f"{file_type.title()} uploaded successfully"
        }
        save_processing_status()

        return {
            "success": True,
            "files": [{
                "video_id": video_id,
                "filename": file.filename,
                "size": total_size,
                "file_type": file_type
            }]
        }

    except Exception as e:
        return {"success": False, "error": f"Upload failed: {str(e)}"}




@app.get("/api/download-recent/{count}")
async def download_recent(count: int):
    """Renvoie le dernier ZIP généré ou le plus récent."""
    zips = sorted(Path("./temp_zips").glob("*.zip"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not zips:
        raise HTTPException(status_code=404, detail="No ZIP found")
    return FileResponse(zips[0], filename=zips[0].name, media_type="application/zip")


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload multiple video/image files for processing"""
    
    if len(files) > 31:
        return {"success": False, "error": "Maximum 30 files allowed"}
    
    results = []
    max_size = 11 * 1024 * 1024  # ✅ 10 MB par fichier
    
    for file in files:
        try:
            if not file.filename:
                continue
                
            file_extension = Path(file.filename).suffix.lower()
            
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.heic', '.heif']
            allowed_extensions = video_extensions + image_extensions
            
            if file_extension not in allowed_extensions:
                continue
            
            file_type = "image" if file_extension in image_extensions else "video"
            video_id = str(uuid.uuid4())
            upload_path = UPLOAD_DIR / f"{video_id}{file_extension}"
            
            total_size = 0
            async with aiofiles.open(upload_path, 'wb') as f:
                while chunk := await file.read(1024 * 1024):  # ✅ 1 MB chunks
                    total_size += len(chunk)
                    if total_size > max_size:
                        await f.close()
                        os.remove(upload_path)
                        break
                    await f.write(chunk)
            
            if 0 < total_size <= max_size:
                processing_status[video_id] = {
                    "status": "uploaded",
                    "filename": file.filename,
                    "file_path": str(upload_path),
                    "file_type": file_type,
                    "progress": 0,
                    "message": f"{file_type.title()} uploaded successfully"
                }
                
                results.append({
                    "video_id": video_id,
                    "filename": file.filename,
                    "size": total_size,
                    "file_type": file_type
                })
        
        except Exception:
            continue
    
    save_processing_status()
    
    return {"success": True, "files": results, "count": len(results)}

from fastapi.responses import FileResponse
import shutil
import tempfile

from fastapi.responses import FileResponse
import tempfile
import shutil

@app.get("/api/download-zip/{video_id}")
async def download_zip(video_id: str):
    """Télécharge le ZIP des fichiers traités et supprime ensuite tout (ZIP + images)."""
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="File not found")

    status = processing_status[video_id]
    zip_path = Path(status.get("zip_path", ""))
    output_path = Path(status.get("output_path", ""))

    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="ZIP not found")

    def cleanup_after_zip():
        """Supprime le ZIP + les fichiers individuels après un délai."""
        try:
            print(f"🕐 Waiting before cleaning ZIP for {video_id}...")
            time.sleep(300)  # 5 min
            # Supprimer ZIP
            if zip_path.exists():
                zip_path.unlink(missing_ok=True)
            # Supprimer dossier output
            folder = output_path.parent if output_path.exists() else None
            if folder and folder.exists():
                shutil.rmtree(folder, ignore_errors=True)
            # Supprimer upload
            for f in Path("uploads").glob(f"{video_id}*"):
                f.unlink(missing_ok=True)
            processing_status.pop(video_id, None)
            save_processing_status()
            print(f"🧹 Cleaned ZIP and related files for {video_id}")
        except Exception as e:
            print(f"⚠️ Cleanup ZIP error for {video_id}: {e}")

    def background_cleanup():
        threading.Thread(target=cleanup_after_zip, daemon=True).start()

    return FileResponse(
        path=zip_path,
        filename=f"sky_results_{video_id}.zip",
        media_type="application/zip",
        background=BackgroundTask(background_cleanup)
    )

@app.get("/api/download-batch-zip/{batch_id}")
async def download_batch_zip(batch_id: str):
    """Télécharge le ZIP global d’un batch."""
    batch_zip = Path(f"./temp_zips/batch_{batch_id}.zip")
    if not batch_zip.exists():
        raise HTTPException(status_code=404, detail="Batch ZIP not found")

    def cleanup_delayed():
        """Supprime le ZIP du batch après 1h."""
        try:
            print(f"🕐 Waiting before cleaning batch ZIP {batch_id}...")
            time.sleep(3600)
            if batch_zip.exists():
                batch_zip.unlink(missing_ok=True)
                print(f"🧹 Cleaned batch ZIP for {batch_id}")
        except Exception as e:
            print(f"⚠️ Cleanup error for batch ZIP {batch_id}: {e}")

    def background_cleanup():
        threading.Thread(target=cleanup_delayed, daemon=True).start()

    return FileResponse(
        path=batch_zip,
        filename=f"SkyAR_batch_{batch_id}.zip",
        media_type="application/zip",
        background=BackgroundTask(background_cleanup)
    )


from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
import tempfile
import shutil
import os

import threading

def periodic_cleanup():
    """Nettoie les fichiers temporaires toutes les heures (ZIP > 1h)."""
    while True:
        try:
            now = time.time()
            # Supprimer les ZIP vieux d'une heure
            temp_zips = Path("temp_zips")
            temp_zips.mkdir(exist_ok=True)
            for zip_file in temp_zips.glob("*.zip"):
                try:
                    if zip_file.stat().st_mtime < now - 3600:  # 1h
                        zip_file.unlink(missing_ok=True)
                        print(f"🧹 Deleted old ZIP: {zip_file.name}")
                except Exception as e:
                    print(f"⚠️ Error deleting old ZIP {zip_file}: {e}")
            time.sleep(3600)  # vérifie toutes les heures
        except Exception as e:
            print(f"⚠️ Cleanup loop error: {e}")


# Lancer le thread en arrière-plan
threading.Thread(target=periodic_cleanup, daemon=True).start()






@app.post("/api/process-batch")
async def process_batch(
    video_ids: List[str] = Form(...),
    randomize_skybox: bool = Form(True),
    sky_template: str = Form(None),
    auto_light_matching: bool = Form(True),
    relighting_factor: float = Form(0.0),
    recoloring_factor: float = Form(0.1),
    halo_effect: bool = Form(True)
):
    """Start batch processing of multiple files sequentially"""
    
    import random

    batch_id = str(uuid.uuid4())
    successful_jobs = []

    # ✅ On traite les fichiers un par un
    for video_id in video_ids:
        if video_id not in processing_status:
            continue

        # Choix du ciel
        if randomize_skybox:
            selected_template = random.choice(list(SKY_TEMPLATES.keys()))
        else:
            selected_template = sky_template or list(SKY_TEMPLATES.keys())[0]

        # Configuration
        request_data = ProcessingRequest(
            video_id=video_id,
            sky_template=selected_template,
            auto_light_matching=auto_light_matching,
            relighting_factor=relighting_factor,
            recoloring_factor=recoloring_factor,
            halo_effect=halo_effect
        )

        # Met à jour le statut
        processing_status[video_id].update({
            "status": "processing",
            "message": f"Processing with {SKY_TEMPLATES[selected_template]['name']} sky...",
            "batch_id": batch_id,
            "selected_skybox": selected_template
        })
        save_processing_status()

        # 🔹 Attendre la fin du traitement avant de passer au suivant
        await run_skyar_processing(video_id, request_data)
        successful_jobs.append(video_id)

    save_processing_status()

    return {
        "success": True,
        "message": f"Batch sequential processing completed for {len(successful_jobs)} files",
        "batch_id": batch_id,
        "jobs": successful_jobs
    }

    """Start batch processing of multiple files"""
    
    import random
    
    batch_id = str(uuid.uuid4())
    successful_jobs = []
    
    for video_id in video_ids:
        if video_id not in processing_status:
            continue
            
        # Assign skybox
        if randomize_skybox:
            selected_template = random.choice(list(SKY_TEMPLATES.keys()))
        else:
            selected_template = sky_template if sky_template else list(SKY_TEMPLATES.keys())[0]
        
        # Create processing request
        request_data = ProcessingRequest(
            video_id=video_id,
            sky_template=selected_template,
            auto_light_matching=auto_light_matching,
            relighting_factor=relighting_factor,
            recoloring_factor=recoloring_factor,
            halo_effect=halo_effect
        )
        
        # Update status
        processing_status[video_id]["status"] = "processing"
        processing_status[video_id]["message"] = f"Processing with {SKY_TEMPLATES[selected_template]['name']} sky..."
        processing_status[video_id]["batch_id"] = batch_id
        processing_status[video_id]["selected_skybox"] = selected_template
        
        # Start processing in background
        asyncio.create_task(run_skyar_processing(video_id, request_data))
        successful_jobs.append(video_id)
    
    save_processing_status()
    
    return {
        "success": True, 
        "message": f"Batch processing started for {len(successful_jobs)} files",
        "batch_id": batch_id,
        "jobs": successful_jobs
    }


@app.get("/api/batch-status/{batch_id}")
async def get_batch_status(batch_id: str):
    """Get status of all files in a batch"""
    
    batch_files = {k: v for k, v in processing_status.items() if v.get("batch_id") == batch_id}
    
    if not batch_files:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    return {
        "batch_id": batch_id,
        "total_files": len(batch_files),
        "completed": len([f for f in batch_files.values() if f["status"] == "completed"]),
        "processing": len([f for f in batch_files.values() if f["status"] == "processing"]),
        "failed": len([f for f in batch_files.values() if f["status"] == "error"]),
        "files": batch_files
    }


@app.post("/api/process")
async def process_video(request: ProcessingRequest):
    """Start video processing with sky replacement"""
    
    video_id = request.video_id
    
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if request.sky_template not in SKY_TEMPLATES:
        raise HTTPException(status_code=400, detail="Invalid sky template")
    
    # Update status
    processing_status[video_id]["status"] = "processing"
    processing_status[video_id]["message"] = "Starting sky replacement..."
    save_processing_status()
    
    # Start processing in background
    asyncio.create_task(run_skyar_processing(video_id, request))
    
    return {"success": True, "message": "Processing started"}


@app.get("/api/status/{video_id}")
async def get_status(video_id: str):
    """Get processing status for a video"""
    
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    status = processing_status[video_id]
    
    # If status shows processing but it's been a while, check if process is still running
    if status.get("status") == "processing":
        # Check if there's an actual skymagic process running for this video
        import subprocess
        try:
            result = subprocess.run(
                ["pgrep", "-f", f"skymagic.py.*{video_id}"],
                capture_output=True, text=True
            )
            if result.returncode != 0:  # No process found
                # Process completed but we missed it - check for output file
                demo_path = Path("./demo.mp4")
                if demo_path.exists():
                    output_dir = OUTPUT_DIR / video_id
                    final_output = output_dir / "result.mp4"
                    if not final_output.exists():
                        # Move the file
                        import shutil
                        shutil.move(str(demo_path), str(final_output))
                    
                    processing_status[video_id].update({
                        "status": "completed",
                        "progress": 100,
                        "message": "Sky replacement completed successfully!",
                        "output_path": str(final_output)
                    })
        except Exception as e:
            # If we can't check the process, just return current status
            pass
    
    return processing_status[video_id]


@app.post("/api/check-completion/{video_id}")
async def check_completion(video_id: str):
    """Manually check if processing is complete and update status"""
    
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Check if output file exists
    demo_path = Path("./demo.mp4")
    if demo_path.exists():
        output_dir = OUTPUT_DIR / video_id
        output_dir.mkdir(exist_ok=True)
        final_output = output_dir / "result.mp4"
        
        if not final_output.exists():
            # Move the file
            import shutil
            shutil.move(str(demo_path), str(final_output))
        
        processing_status[video_id].update({
            "status": "completed", 
            "progress": 100,
            "message": "Sky replacement completed successfully!",
            "output_path": str(final_output)
        })
        
        return {"success": True, "message": "Processing completed"}
    else:
        return {"success": False, "message": "Still processing"}


from starlette.background import BackgroundTask
import threading, time

@app.get("/api/download/{video_id}")
async def download_result(video_id: str):
    """Télécharge le fichier traité et nettoie uniquement ce fichier après un délai (sans casser le ZIP)."""
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="File not found")

    status = processing_status[video_id]
    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed")

    output_path = Path(status.get("output_path"))
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Processed file not found")

    file_type = status.get("file_type", "video")
    filename = f"skyar_result_{video_id}.{'jpg' if file_type == 'image' else 'mp4'}"
    media_type = "image/jpeg" if file_type == "image" else "video/mp4"

    # ✅ S'assurer que le ZIP existe encore (le régénérer si supprimé)
    zip_path = Path(status.get("zip_path", ""))
    output_dir = output_path.parent
    try:
        if not zip_path.exists():
            zip_dir = Path("./temp_zips")
            zip_dir.mkdir(exist_ok=True)
            new_zip = zip_dir / f"{video_id}.zip"
            shutil.make_archive(str(new_zip.with_suffix("")), "zip", output_dir)
            processing_status[video_id]["zip_path"] = str(new_zip)
            save_processing_status()
            print(f"♻️ Recreated missing ZIP for {video_id}")
    except Exception as e:
        print(f"⚠️ Could not recreate ZIP for {video_id}: {e}")

    def cleanup_delayed():
        """Supprime uniquement le fichier individuel (pas le ZIP)."""
        try:
            print(f"🕐 Waiting before cleaning single file for {video_id}...")
            time.sleep(300)  # 5 minutes
            # Supprime uniquement le fichier (pas le dossier ni le ZIP)
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            print(f"🧹 Cleaned single file for {video_id}")
        except Exception as e:
            print(f"⚠️ Cleanup error for {video_id}: {e}")

    # Exécuter le nettoyage dans un thread séparé
    def background_cleanup():
        threading.Thread(target=cleanup_delayed, daemon=True).start()

    return FileResponse(
        path=output_path,
        filename=filename,
        media_type=media_type,
        background=BackgroundTask(background_cleanup)
    )




@app.get("/api/skybox/{filename}")
async def get_skybox_image(filename: str):
    """Serve skybox images for previews"""
    skybox_path = Path("./skybox") / filename
    if skybox_path.exists() and skybox_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        return FileResponse(path=skybox_path, media_type="image/jpeg")
    else:
        raise HTTPException(status_code=404, detail="Skybox image not found")


@app.get("/api/templates")
async def get_sky_templates():
    """Get available sky templates"""
    return {"templates": SKY_TEMPLATES}


@app.post("/api/upload-skybox")
async def upload_skybox(file: UploadFile = File(...), skybox_name: str = Form(...)):
    """Upload a new skybox image"""
    
    try:
        # Validate skybox name
        valid_names = ["bluesky1", "bluesky2", "bluesky3", "bluesky4"]
        if skybox_name not in valid_names:
            return {"success": False, "error": f"Invalid skybox name. Must be one of: {valid_names}"}
        
        # Validate file type
        if not file.filename:
            return {"success": False, "error": "No file provided"}
            
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ['.jpg', '.jpeg', '.png']:
            return {"success": False, "error": "Only JPG and PNG files are allowed"}
        
        # Save to skybox directory
        skybox_path = Path("./skybox") / f"{skybox_name}.jpg"
        
        async with aiofiles.open(skybox_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        return {
            "success": True,
            "message": f"Skybox {skybox_name} uploaded successfully",
            "filename": file.filename,
            "size": len(content)
        }
    
    except Exception as e:
        return {"success": False, "error": f"Upload failed: {str(e)}"}


@app.get("/upload-skybox", response_class=HTMLResponse)
async def upload_skybox_page(request: Request):
    """Serve the skybox upload page"""
    return templates.TemplateResponse("upload_skybox.html", {"request": request})


@app.get("/upload-test", response_class=HTMLResponse)
async def upload_test_page(request: Request):
    """Serve the upload test page for diagnostics"""
    return templates.TemplateResponse("upload_test.html", {"request": request})


@app.get("/test", response_class=HTMLResponse)
async def test_page(request: Request):
    """Serve a simple test page"""
    return templates.TemplateResponse("test.html", {"request": request})


async def run_skyar_processing(video_id: str, request: ProcessingRequest):
    """Run SkyAR processing in the background for both images and videos."""
    try:
        status = processing_status[video_id]
        input_path = Path(status["file_path"])
        file_type = status.get("file_type", "video")

        # ========== Préparation de l’entrée ==========
        if file_type == "image":
            from PIL import Image
            import shutil

            input_dir = Path(f"./temp_image_input_{video_id}")
            input_dir.mkdir(exist_ok=True)

            img = Image.open(input_path)
            orig_width, orig_height = img.size
            max_4k_width, max_4k_height = 3840, 2160

            if orig_width <= max_4k_width and orig_height <= max_4k_height:
                out_width, out_height = orig_width, orig_height
                print(f"Keeping original resolution: {out_width}x{out_height}")
            else:
                aspect_ratio = orig_width / orig_height
                if aspect_ratio > (max_4k_width / max_4k_height):
                    out_width = max_4k_width
                    out_height = int(max_4k_width / aspect_ratio)
                else:
                    out_height = max_4k_height
                    out_width = int(max_4k_height * aspect_ratio)
                print(f"Scaling from {orig_width}x{orig_height} to {out_width}x{out_height}")

            out_width -= out_width % 2
            out_height -= out_height % 2

            shutil.copy2(input_path, input_dir / input_path.name)
            input_mode = "seq"
            datadir = str(input_dir.resolve())

        else:  # VIDÉO
            input_mode = "video"
            datadir = str(input_path.resolve())
            out_width, out_height = 640, 360  # Résolution réduite pour vitesse

        # ========== Configuration du traitement ==========
        if file_type == "image":
            if out_width >= 2560 or out_height >= 1440:
                skybox_center_crop = 0.8
                in_size_w, in_size_h = 512, 512
            else:
                skybox_center_crop = 0.6
                in_size_w, in_size_h = 384, 384
        else:
            skybox_center_crop = 0.5
            in_size_w, in_size_h = 384, 384

        output_dir = OUTPUT_DIR / video_id
        output_dir.mkdir(exist_ok=True)

        config = {
            "net_G": "coord_resnet50",
            "ckptdir": str((BASE_DIR / "checkpoints_G_coord_resnet50").resolve()),
            "input_mode": input_mode,
            "datadir": datadir,
            "skybox": SKY_TEMPLATES[request.sky_template]["file"],
            "in_size_w": in_size_w,
            "in_size_h": in_size_h,
            "out_size_w": out_width,
            "out_size_h": out_height,
            "skybox_center_crop": skybox_center_crop,
            "auto_light_matching": request.auto_light_matching,
            "relighting_factor": request.relighting_factor,
            "recoloring_factor": request.recoloring_factor,
            "halo_effect": request.halo_effect,
            "output_dir": str(output_dir.resolve()),
            "save_jpgs": file_type == "image"
        }

        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # ========== Lancement du traitement ==========
        import sys
        python_exec = sys.executable
        skymagic_path = str(BASE_DIR / "skymagic.py")

        cmd = [python_exec, skymagic_path, "--path", str(config_path.resolve())]
        print(f"[DEBUG] Running: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(BASE_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        async def monitor_progress():
            """Mise à jour du pourcentage de progression."""
            steps = [25, 50, 75, 90] if file_type == "image" else [15, 30, 45, 60, 75, 85, 92, 95]
            interval = 3 if file_type == "image" else 15
            for p in steps:
                if process.returncode is not None:
                    break
                await asyncio.sleep(interval)
                processing_status[video_id].update({
                    "progress": p,
                    "message": f"Processing {file_type}... ({p}%)"
                })
                save_processing_status()

        monitor_task = asyncio.create_task(monitor_progress())
        stdout, stderr = await process.communicate()
        monitor_task.cancel()

        # ========== Vérification du résultat ==========
        if process.returncode != 0:
            error_message = stderr.decode("utf-8", errors="replace")
            raise Exception(f"SkyAR failed: {error_message}")

        # Sortie finale
        if file_type == "image":
            import glob
            candidates = list(output_dir.glob("*syneth.jpg"))
            if not candidates:
                candidates = [f for f in output_dir.glob("*.jpg") if "mask" not in f.name.lower()]
            if not candidates:
                raise Exception("No output image found.")
            final_output = output_dir / "result.jpg"
            shutil.copy2(candidates[0], final_output)
        else:  # vidéo
            demo_path = Path("./demo.mp4")
            if not demo_path.exists():
                raise Exception("No output video generated.")
            final_output = output_dir / "result.mp4"
            shutil.move(demo_path, final_output)

        # ✅ Mise à jour du statut
        processing_status[video_id].update({
            "status": "completed",
            "progress": 100,
            "message": "Sky replacement completed successfully!",
            "output_path": str(final_output)
        })
        save_processing_status()

        # ✅ Création automatique du ZIP
        try:
            zip_dir = Path("./temp_zips")
            zip_dir.mkdir(exist_ok=True)
            zip_path = zip_dir / f"{video_id}.zip"
            shutil.make_archive(str(zip_path.with_suffix("")), "zip", output_dir)
            processing_status[video_id]["zip_path"] = str(zip_path)
            save_processing_status()
            print(f"📦 ZIP créé pour {video_id} → {zip_path}")
        except Exception as e:
            print(f"⚠️ Erreur lors de la création du ZIP pour {video_id}: {e}")

        # ✅ Nettoyage temporaire
        if file_type == "image" and "input_dir" in locals():
            shutil.rmtree(input_dir, ignore_errors=True)

        print(f"✅ Processing completed for {video_id}: {final_output}")

    except Exception as e:
        # ❌ En cas d’erreur
        processing_status[video_id].update({
            "status": "error",
            "progress": 0,
            "message": f"Processing failed: {str(e)}"
        })
        save_processing_status()
        print(f"❌ Processing failed for {video_id}: {e}")



# =============================
# 🧹 Nettoyage automatique périodique avec logs persistants
# =============================
import threading
import time
import shutil
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

# Intervalle de vérification (secondes)
CLEANUP_INTERVAL = 30 * 60   # Par défaut : 30 minutes
# Âge maximum des fichiers (secondes)
FILE_EXPIRATION = 60 * 60    # Par défaut : 1 heure

# === Configuration du logger ===
LOG_FILE = Path(BASE_DIR) / "cleanup.log"
handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3)
logging.basicConfig(
    handlers=[handler],
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("cleanup")

def auto_cleanup():
    """Supprime périodiquement les fichiers anciens dans uploads/, outputs/ et temp_zips/."""
    paths = [Path("uploads"), Path("outputs"), Path("temp_zips")]

    while True:
        try:
            now = time.time()
            logger.info("🧹 Vérification des fichiers anciens...")

            for folder in paths:
                folder.mkdir(exist_ok=True)
                for item in folder.iterdir():
                    try:
                        age_seconds = now - item.stat().st_mtime
                        if age_seconds > FILE_EXPIRATION:
                            if item.is_dir():
                                shutil.rmtree(item, ignore_errors=True)
                                logger.info(f"🗑️ Dossier supprimé (vieux de {age_seconds/60:.1f} min) : {item}")
                            else:
                                item.unlink(missing_ok=True)
                                logger.info(f"🗑️ Fichier supprimé (vieux de {age_seconds/60:.1f} min) : {item}")
                    except Exception as e:
                        logger.warning(f"⚠️ Erreur de suppression {item}: {e}")

            logger.info(f"✅ Nettoyage terminé. Prochaine vérification dans {CLEANUP_INTERVAL/60:.0f} min.\n")
            time.sleep(CLEANUP_INTERVAL)

        except Exception as e:
            logger.error(f"⚠️ Erreur dans la boucle de nettoyage : {e}")
            time.sleep(CLEANUP_INTERVAL)

# Lancer le thread en arrière-plan
threading.Thread(target=auto_cleanup, daemon=True).start()




@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "SkyAR Demo"}

@app.get("/api/download-latest-results")
async def download_latest_results():
    """
    Cherche tous les outputs terminés, récupère leurs result.jpg
    puis crée un ZIP contenant uniquement ces images finales.
    """

    # Trouver tous les result.jpg existants dans outputs/*
    result_images = list(Path("outputs").glob("*/result.jpg"))

    if not result_images:
        raise HTTPException(status_code=404, detail="No processed images found yet")

    # Créer un ZIP temporaire
    TEMP_ZIPS_DIR.mkdir(exist_ok=True)
    zip_path = TEMP_ZIPS_DIR / f"sky_results_{int(time.time())}.zip"

    import zipfile
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for i, img_path in enumerate(result_images):
            # Nom dans le zip → result_1.jpg, result_2.jpg...
            zipf.write(img_path, arcname=f"result_{i+1}.jpg")

    return FileResponse(
        path=zip_path,
        filename=zip_path.name,
        media_type="application/zip"
    )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)