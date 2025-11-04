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
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(title="SkyAR Demo", description="Dynamic Sky Replacement in Videos")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for uploads and outputs (local paths)
UPLOAD_DIR = Path("./uploads")
OUTPUT_DIR = Path("./outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Mount static files and templates
templates = Jinja2Templates(directory="templates")

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


@app.post("/api/upload-single")
async def upload_single_file(file: UploadFile = File(...)):
    """Upload a single file (for backward compatibility)"""
    
    try:
        # Validate file type
        if not file.filename:
            return {"success": False, "error": "No file provided"}
            
        file_extension = Path(file.filename).suffix.lower()
        
        # Define allowed file types
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.heic', '.heif']
        allowed_extensions = video_extensions + image_extensions
        
        if file_extension not in allowed_extensions:
            return {"success": False, "error": f"Unsupported file type. Please upload a video ({', '.join(video_extensions)}) or image ({', '.join(image_extensions)}) file."}
        
        # Determine file type
        is_image = file_extension in image_extensions
        is_video = file_extension in video_extensions
        
        # Set appropriate size limits (4MB confirmed working in tests)
        if is_image:
            max_size = 4 * 1024 * 1024  # 4MB for images
            file_type = "image"
        else:
            max_size = 4 * 1024 * 1024  # 4MB for videos
            file_type = "video"
        
        # Generate unique ID for this upload
        video_id = str(uuid.uuid4())
        
        # Save uploaded file
        upload_path = UPLOAD_DIR / f"{video_id}{file_extension}"
        
        async with aiofiles.open(upload_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Validate file size based on type
        if len(content) == 0:
            return {"success": False, "error": "Uploaded file is empty"}
        
        if len(content) > max_size:
            size_limit = "4MB"
            return {"success": False, "error": f"File too large. Maximum size for {file_type}s is {size_limit}"}
        
        # Initialize processing status
        processing_status[video_id] = {
            "status": "uploaded",
            "filename": file.filename,
            "file_path": str(upload_path),
            "file_type": file_type,
            "progress": 0,
            "message": f"{file_type.title()} uploaded successfully"
        }
        
        results = [{
            "video_id": video_id,
            "filename": file.filename,
            "size": len(content),
            "file_type": file_type
        }]
    
        save_processing_status()
        
        return {
            "success": True,
            "files": results,
            "count": len(results)
        }
    
    except Exception as e:
        return {"success": False, "error": f"Upload failed: {str(e)}"}


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload multiple video/image files for processing"""
    
    if len(files) > 10:
        return {"success": False, "error": "Maximum 10 files allowed"}
    
    results = []
    
    for file in files:
        try:
            # Validate file type
            if not file.filename:
                continue
                
            file_extension = Path(file.filename).suffix.lower()
            
            # Define allowed file types
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.heic', '.heif']
            allowed_extensions = video_extensions + image_extensions
            
            if file_extension not in allowed_extensions:
                continue
            
            # Determine file type
            is_image = file_extension in image_extensions
            is_video = file_extension in video_extensions
            
            # Set appropriate size limits (4MB confirmed working in tests)
            if is_image:
                max_size = 4 * 1024 * 1024  # 4MB for images
                file_type = "image"
            else:
                max_size = 4 * 1024 * 1024  # 4MB for videos
                file_type = "video"
            
            # Generate unique ID for this upload
            video_id = str(uuid.uuid4())
            
            # Save uploaded file
            upload_path = UPLOAD_DIR / f"{video_id}{file_extension}"
            
            async with aiofiles.open(upload_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Validate file size based on type
            if len(content) == 0:
                continue
            
            if len(content) > max_size:
                continue
            
            # Initialize processing status
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
                "size": len(content),
                "file_type": file_type
            })
        
        except Exception as e:
            continue
    
    save_processing_status()
    
    return {
        "success": True,
        "files": results,
        "count": len(results)
    }


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


@app.get("/api/download/{video_id}")
async def download_result(video_id: str):
    """Download the processed file (image or video)"""
    
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="File not found")
    
    status = processing_status[video_id]
    
    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed")
    
    output_path = Path(status.get("output_path"))
    
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Processed file not found")
    
    # Determine file type and set appropriate response
    file_type = status.get("file_type", "video")
    if file_type == "image":
        media_type = "image/jpeg"
        filename = f"skyar_result_{video_id}.jpg"
    else:
        media_type = "video/mp4"
        filename = f"skyar_result_{video_id}.mp4"
    
    return FileResponse(
        path=output_path,
        filename=filename,
        media_type=media_type
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
    """Run SkyAR processing in the background for both images and videos"""
    
    try:
        status = processing_status[video_id]
        input_path = status["file_path"]
        file_type = status.get("file_type", "video")  # Default to video for backward compatibility
        
        # Determine input mode and output size based on file type
        if file_type == "image":
            # For images, we need to create a directory and copy the image there
            # because skymagic expects a directory for 'seq' mode
            input_dir = Path(f"./temp_image_input_{video_id}")
            input_dir.mkdir(exist_ok=True)
            
            # Copy image to the input directory
            import shutil
            from PIL import Image
            
            # Get original image dimensions to maintain aspect ratio
            img = Image.open(input_path)
            orig_width, orig_height = img.size
            
            # Calculate output size while maintaining aspect ratio (max 4K: 3840x2160)
            max_4k_width, max_4k_height = 3840, 2160
            
            if orig_width <= max_4k_width and orig_height <= max_4k_height:
                # Image is 4K or smaller - keep original resolution
                out_width = orig_width
                out_height = orig_height
                print(f"Keeping original resolution: {out_width}x{out_height}")
            else:
                # Image is larger than 4K - scale down while preserving aspect ratio
                aspect_ratio = orig_width / orig_height
                
                if aspect_ratio > (max_4k_width / max_4k_height):
                    # Width is the limiting factor
                    out_width = max_4k_width
                    out_height = int(max_4k_width / aspect_ratio)
                else:
                    # Height is the limiting factor
                    out_height = max_4k_height
                    out_width = int(max_4k_height * aspect_ratio)
                
                print(f"Scaling from {orig_width}x{orig_height} to {out_width}x{out_height} (4K max)")
            
            # Ensure dimensions are even numbers (required for video processing)
            out_width = out_width if out_width % 2 == 0 else out_width - 1
            out_height = out_height if out_height % 2 == 0 else out_height - 1
            
            image_name = Path(input_path).name
            shutil.copy2(input_path, input_dir / image_name)
            
            input_mode = "seq"
            datadir = str(input_dir)
        else:
            input_mode = "video"
            datadir = input_path
            out_width, out_height = 640, 360  # Lower resolution for videos

        # Optimize skybox quality based on output resolution
        if file_type == "image":
            # For high-resolution images, adjust skybox processing for better quality
            if out_width >= 2560 or out_height >= 1440:  # 2K+ images
                skybox_center_crop = 0.8  # Less cropping for high-res
                in_size_w, in_size_h = 512, 512  # Higher processing resolution
            else:
                skybox_center_crop = 0.6  # Standard cropping
                in_size_w, in_size_h = 384, 384  # Standard processing resolution
        else:
            # For videos, use standard settings
            skybox_center_crop = 0.5
            in_size_w, in_size_h = 384, 384
        
        # Create temporary config file
        config = {
            "net_G": "coord_resnet50",
            "ckptdir": "./checkpoints_G_coord_resnet50",
            "input_mode": input_mode,
            "datadir": datadir,
            "skybox": SKY_TEMPLATES[request.sky_template]["file"],
            "in_size_w": in_size_w,
            "in_size_h": in_size_h,
            "out_size_w": out_width,
            "out_size_h": out_height,
            "skybox_center_crop": skybox_center_crop,  # Optimized cropping
            "auto_light_matching": request.auto_light_matching,
            "relighting_factor": request.relighting_factor,
            "recoloring_factor": request.recoloring_factor,
            "halo_effect": request.halo_effect,
            "output_dir": str(OUTPUT_DIR / video_id),
            "save_jpgs": True if file_type == "image" else False  # Save JPGs for images
        }
        
        # Create output directory
        output_dir = OUTPUT_DIR / video_id
        output_dir.mkdir(exist_ok=True)
        
        # Save config file
        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Update status
        processing_status[video_id].update({
            "status": "processing",
            "progress": 10,
            "message": f"Processing {file_type} with {SKY_TEMPLATES[request.sky_template]['name']} sky..."
        })
        save_processing_status()  # Save to file
        
        # Run SkyAR processing
        cmd = [
            "python3", "skymagic.py",
            "--path", str(config_path)
        ]
        
        # Change to current directory and run
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=".",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Monitor the process and update progress periodically
        start_time = time.time()
        
        # Update progress while process is running
        async def monitor_progress():
            if file_type == "image":
                # Images process much faster
                progress_values = [25, 50, 75, 90]
                base_interval = 3  # 3 seconds for images
            else:
                # Videos take longer
                progress_values = [15, 30, 45, 60, 75, 85, 92, 95, 97, 98]
                base_interval = 15  # 15 seconds for videos
            
            for i, progress in enumerate(progress_values):
                if process.returncode is not None:
                    break
                    
                wait_time = base_interval if file_type == "image" else (base_interval if i < 6 else base_interval * 2)
                await asyncio.sleep(wait_time)
                
                if video_id in processing_status:  # Check if still exists
                    processing_status[video_id].update({
                        "progress": progress,
                        "message": f"Processing {file_type}... ({progress}%)"
                    })
                    save_processing_status()  # Save to file
            
            # If we've gone through all progress steps but process is still running,
            # keep it at high percentage and check periodically
            final_progress = 95 if file_type == "image" else 98
            while process.returncode is None:
                await asyncio.sleep(5 if file_type == "image" else 10)
                if video_id in processing_status:
                    processing_status[video_id].update({
                        "progress": final_progress,
                        "message": f"Finalizing {file_type} processing..."
                    })
                    save_processing_status()  # Save to file
        
        # Start progress monitoring
        progress_task = asyncio.create_task(monitor_progress())
        
        # Wait for process to complete
        stdout, stderr = await process.communicate()
        
        # Cancel progress monitoring
        progress_task.cancel()
        
        if process.returncode == 0:
            # Processing completed successfully
            if file_type == "image":
                # For images, look for the processed image in the output directory
                # The filename pattern should be: imagename_syneth.jpg (without extension)
                import glob
                
                # Find syneth files
                syneth_files = glob.glob(str(output_dir / "*syneth.jpg"))
                
                if syneth_files:
                    # Use the first syneth file found
                    final_output = output_dir / "result.jpg"
                    shutil.copy2(syneth_files[0], str(final_output))
                else:
                    # Debug: list all files in output directory
                    all_files = list(output_dir.glob("*"))
                    print(f"Debug: Files in output directory: {[f.name for f in all_files]}")
                    
                    # Try to find any processed image (not input or mask)
                    candidates = [f for f in all_files 
                                if f.suffix.lower() in ['.jpg', '.jpeg'] 
                                and not any(x in f.name.lower() for x in ['input', 'mask', 'config'])]
                    
                    if candidates:
                        # Sort by size (largest is likely the result)
                        candidates.sort(key=lambda x: x.stat().st_size, reverse=True)
                        final_output = output_dir / "result.jpg"
                        shutil.copy2(str(candidates[0]), str(final_output))
                    else:
                        raise Exception(f"Output image not generated - found files: {[f.name for f in all_files]}")
            else:
                # For videos, look for demo.mp4
                demo_path = Path("./demo.mp4")
                if demo_path.exists():
                    final_output = output_dir / "result.mp4"
                    shutil.move(str(demo_path), str(final_output))
                else:
                    raise Exception("Output video not generated")
            
            processing_status[video_id].update({
                "status": "completed",
                "progress": 100,
                "message": f"Sky replacement completed successfully!",
                "output_path": str(final_output)
            })
            save_processing_status()  # Save to file
            
            # Clean up temporary image directory if it was created
            if file_type == "image" and 'input_dir' in locals():
                import shutil
                shutil.rmtree(input_dir, ignore_errors=True)
                
        else:
            raise Exception(f"Processing failed: {stderr.decode()}")
    
    except Exception as e:
        processing_status[video_id].update({
            "status": "error",
            "progress": 0,
            "message": f"Processing failed: {str(e)}"
        })
        save_processing_status()  # Save to file


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "SkyAR Demo"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)