"""
Resound Studio - FastAPI Backend
============================
Main server exposing REST API endpoints for:
  - Voice cloning (upload audio → extract embedding → save)
  - Voice design (text description → embedding → save)
  - TTS generation (text + voice + settings → audio)
  - Voice management (list, delete, preview)
  - Generation history (list, replay, delete)
  - Voice profiles (CRUD, multi-sample support)

Run with:
  uvicorn main:app --reload --port 8000
"""

import json
import logging
import os
import gc
import uuid
import time
import threading
import io
import os
from contextlib import asynccontextmanager

# Suppress noisy SoX/Audio warnings early
os.environ["SOX_VERBOSITY"] = "0"

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse, StreamingResponse, JSONResponse
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

# ---- New modular imports ----
from database import init_db, get_db
import profiles as profiles_module
import history as history_module
import stories as stories_module
import channels as channels_module
from schemas import (
    GenerateRequest, DesignVoiceRequest, PreviewRequest,
    LoadModelRequest, PodcastTimelineRequest,
    StreamGenerateRequest, AsyncGenerateRequest,
    BatchItem, BatchGenerateRequest,
    ConversationRequest, AudiobookRequest,
    EmotionSegment, EmotionTimelineRequest,
    CompareRequest, ConvertRequest, SrtRequest,
    VoiceProfileCreate, VoiceProfileUpdate,
    PreviewDesignVoiceRequest, 
    HistoryFilters,
    StoryCreate, StoryItemCreate, StoryItemMove, StoryItemTrim, 
    StoryResponse, StoryItemResponse,
)

# ---- Legacy imports (engine & utilities — kept as-is) ----
from engine_manager import get_manager, detect_accelerators
from utils.audio_utils import (
    master_audio, sanitize_reference_audio,
    validate_reference_audio, normalize_audio, load_audio_bytes,
)
from utils.text_chunker import chunk_text
from utils.voice_prompt_cache import get_prompt_cache
from utils.progress import get_progress_manager
from utils.audio_cache import get_cached, put_cached, clear_cache as clear_audio_cache, get_cache_stats
from utils.features import (
    parse_multi_speaker_script, generate_multi_speaker_audio,
    split_into_chapters, detect_dialogue,
    export_voice, import_voice,
    generate_srt, estimate_segment_timing,
    mix_audio_with_music,
    parse_emotion_timeline,
    convert_audio_format,
)
from utils.api_key_middleware import ApiKeyMiddleware, API_KEY

# ---- Logging ----
# Use simple StreamHandler for Docker logs unless explicitly configured for file logging
IS_DOCKER = os.environ.get("IS_DOCKER", "false").lower() == "true"
log_handlers = [logging.StreamHandler()]

if not IS_DOCKER:
    log_file_path = os.path.join(os.path.dirname(__file__), "data", "resound-studio.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    log_handlers.append(logging.FileHandler(log_file_path, encoding="utf-8"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=log_handlers
)
logger = logging.getLogger("resound-studio")

# ---- Lifespan ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Existing lifespan code...
    logger.info("=" * 60)
    logger.info("  Resound Studio API Server Starting")
    logger.info("  Multi-Model Architecture v2.0")
    logger.info("=" * 60)

    # ---- Initialize database ----
    init_db()
    logger.info("Database initialized (SQLite + SQLAlchemy)")

    # ---- Migrate legacy flat-file voices into the DB ----
    try:
        db = next(get_db())
        migrated = profiles_module.migrate_from_voice_store(db)
        if migrated > 0:
            logger.info(f"Migrated {migrated} legacy voices into database")
        db.close()
    except Exception as e:
        logger.warning(f"Legacy migration skipped: {e}")
    
    # Enable automatic CuDNN optimizations for generation algorithms
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled CuDNN benchmarks for optimized performance.")

            # Check for flash-attn (#3)
            try:
                import flash_attn
                logger.info(f"flash-attn v{flash_attn.__version__} detected ✓ (2-4x faster attention)")
            except ImportError:
                logger.warning(
                    "flash-attn is NOT installed. Install it for 2-4x faster Qwen inference: "
                    "pip install flash-attn --no-build-isolation"
                )
    except ImportError:
        pass

    # ---- Initialize progress manager with event loop ----
    try:
        import asyncio
        loop = asyncio.get_running_loop()
        progress_mgr = get_progress_manager()
        progress_mgr.set_event_loop(loop)
        logger.info("Progress manager initialized with event loop")
    except Exception as e:
        logger.warning(f"Progress manager init: {e}")

    # ---- Detect accelerators ----
    try:
        accel = detect_accelerators()
        rec = accel['recommended']
        cuda_avail = accel['cuda']['available']
        
        logger.info(f"HARDWARE LOG: Recommended={rec}, CUDA={cuda_avail}")
        
        if not cuda_avail:
            # Check if an NVIDIA card is actually present but invisible to torch
            import subprocess
            try:
                smi = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
                logger.warning("CRITICAL: NVIDIA-SMI detected a GPU, but PyTorch cannot see it! Using CPU fallback.")
                logger.warning("HINT: You might be using the CPU-only version of torch. Try: pip install torch --index-url https://download.pytorch.org/whl/cu121")
            except:
                pass
    except Exception as e:
        logger.warning(f"Accelerator detection error: {e}")
        
    yield
    logger.info("Resound Studio API Server Shutting Down")

# ---- App ----
app = FastAPI(
    title="Resound Studio API",
    description="Multi-Model Zero-Shot Voice Synthesis System",
    version="2.0.0",
    lifespan=lifespan,
)

# ---- Global Exception Handler ----
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = str(exc)
    logger.error(f"GLOBAL ERROR: {error_msg}", exc_info=True)
    
    # Check for OOM
    if "out of memory" in error_msg.lower() or "CUDA out of memory" in error_msg:
        logger.warning("CRITICAL: GPU OOM detected. Clearing cache and unloading active model...")
        manager = get_manager()
        if manager.active_model_id:
            manager.unload_model(manager.active_model_id)
            
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "A critical system error occurred.",
            "detail": error_msg,
            "code": "INTERNAL_SERVER_ERROR"
        }
    )


# CORS for Next.js frontend
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True if "*" not in allowed_origins else False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key middleware (only active when RESOUND_API_KEY env var is set)
if API_KEY:
    app.add_middleware(ApiKeyMiddleware)
    logger.info(f"API Key authentication ENABLED (rate limit: {os.environ.get('RESOUND_RATE_LIMIT', '60')}/min)")
else:
    logger.info("API Key authentication DISABLED (set RESOUND_API_KEY to enable)")

# Register Modular Routers
app.include_router(channels_module.router)



# ---- Models are now imported from schemas.py ----

# ---- Backward-compatible helper for routes that don't yet take db: Session ----
# These wrap the profiles_module to work without explicit DI for background threads.
def _get_voice_compat(voice_id: str):
    """Backward-compat wrapper: get voice profile using a transient session."""
    from database import get_session
    db = get_session()
    try:
        return profiles_module.get_profile(voice_id, db)
    finally:
        db.close()

def _get_embedding_path_compat(voice_id: str):
    """Backward-compat wrapper: get embedding path using a transient session."""
    from database import get_session
    db = get_session()
    try:
        return profiles_module.get_profile_embedding_path(voice_id, db)
    finally:
        db.close()

def _get_sample_path_compat(voice_id: str):
    """Backward-compat wrapper: get sample path using a transient session."""
    from database import get_session
    db = get_session()
    try:
        return profiles_module.get_profile_sample_path(voice_id, db)
    finally:
        db.close()


# ---- Health ----
@app.get("/")
async def root():
    return {"status": "ok", "service": "Resound Studio API", "model": "Qwen3-TTS-1.7B-INT8"}


@app.get("/health")
async def health():
    manager = get_manager()
    try:
        accelerators = manager.get_accelerator_info()
        device = accelerators.get("recommended", "cpu")
    except Exception as e:
        logger.warning(f"Accelerator detection failed: {e}")
        accelerators = {}
        device = "cpu"
    return {
        "status": "ok",
        "active_model": manager.active_model_id,
        "device": device,
        "accelerators": accelerators,
    }

# ==============================
# Model Management
# ==============================
@app.get("/api/models")
async def list_models():
    """List all available AI audio models and their current load status."""
    manager = get_manager()
    models = manager.get_available_models()
    return {
        "active": manager.active_model_id,
        "models": models
    }

# LoadModelRequest imported from schemas.py

@app.post("/api/models/load")
async def load_model_endpoint(req: LoadModelRequest):
    """Dynamically switch and load an engine into VRAM."""
    manager = get_manager()
    try:
        engine = manager.load_model(req.model_id)
        return {"status": "success", "model": req.model_id, "device": engine.device}
    except Exception as e:
        logger.error(f"Failed to load model {req.model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/unload")
async def unload_model_endpoint(req: LoadModelRequest):
    """Explicitly unload a model and free VRAM."""
    manager = get_manager()
    try:
        success = manager.unload_model(req.model_id)
        if success:
            return {"status": "success", "message": f"Model {req.model_id} unloaded."}
        else:
            return {"status": "error", "message": f"Model {req.model_id} was not loaded."}
    except Exception as e:
        logger.error(f"Failed to unload model {req.model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/load-stream")
async def load_model_stream(model_id: str):
    """
    SSE endpoint that streams real-time progress while loading a model.
    Usage: GET /api/models/load-stream?model_id=qwen-1.7b
    """
    manager = get_manager()

    def event_generator():
        for progress in manager.load_model_with_progress(model_id):
            yield progress.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ==============================
# Server Logs
# ==============================
@app.get("/api/logs")
async def get_logs():
    """Retrieve the recent backend logs."""
    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return {"logs": "".join(lines[-1000:])}
    except Exception as e:
        return {"logs": f"Failed to read logs: {e}"}

# ==============================
# Voice Cloning
# ==============================
@app.post("/api/clone")
async def clone_voice(
    audio: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(""),
    language: str = Form("English"),
    tags: str = Form("[]"),
    reference_text: str = Form(""),
    db: Session = Depends(get_db),
):
    """Clone a voice from an uploaded audio sample.
    
    Phase 0 improvements:
      - Validates audio (duration 2-30s, volume, clipping)
      - Normalizes audio before cloning
      - Uses reference_text for full phoneme alignment (auto-transcribes if empty)
    """
    logger.info(f"Cloning voice: {name} (file: {audio.filename})")

    # Read audio bytes
    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(400, "Empty audio file")

    # Phase 0A: Validate reference audio
    is_valid, error_msg = validate_reference_audio(audio_bytes=audio_bytes)
    if not is_valid:
        logger.warning(f"Reference audio validation: {error_msg}")
        # Don't hard-fail for backward compatibility, but log warning
        
    # Remove background noise to improve clone quality
    audio_bytes = sanitize_reference_audio(audio_bytes)

    # Parse tags
    try:
        tag_list = json.loads(tags) if tags else []
    except json.JSONDecodeError:
        tag_list = []

    manager = get_manager()

    # If it's a designed voice, its pure text description is all we need as the "prompt"
    if "designed" in tag_list:
        embedding_bytes = description.encode('utf-8')
    else:
        # Clone voice using the active manager engine
        engine = manager.get_current_engine()
        
        # Phase 0B: Pass reference_text for full phoneme alignment
        clone_result = engine.clone_voice(audio_bytes, ref_text=reference_text)
        embedding_bytes = clone_result["prompt_bytes"]
        
        # Use auto-transcribed text if engine returned it
        if not reference_text and "reference_text" in clone_result:
            reference_text = clone_result["reference_text"]

    # Create profile in database
    profile_data = VoiceProfileCreate(
        name=name,
        description=description,
        language=language,
        tags=tag_list,
    )
    profile = profiles_module.create_profile(
        data=profile_data,
        engine_id=manager.active_model_id or "unknown",
        db=db,
    )

    # Add the audio sample to the profile
    duration = None
    try:
        data_arr, sr = sf.read(io.BytesIO(audio_bytes))
        duration = len(data_arr) / sr
    except Exception:
        pass

    profiles_module.add_sample(
        profile_id=profile.id,
        audio_bytes=audio_bytes,
        embedding_bytes=embedding_bytes,
        reference_text=reference_text,
        duration_seconds=duration,
        is_primary=True,
        db=db,
    )

    result = profiles_module._profile_to_dict(profile, sample_count=1)
    logger.info(f"Voice cloned successfully: {profile.id}")
    return result


# ==============================
# Voice Design
# ==============================
@app.post("/api/design-voice")
async def design_voice(req: DesignVoiceRequest, db: Session = Depends(get_db)):
    """Create a new voice from a text description."""
    logger.info(f"Designing voice: {req.name}")

    manager = get_manager()
    engine = manager.get_current_engine()

    # Generate design audio and use it to create a clone prompt
    design_audio = engine.design_voice(
        description=req.description,
        text="Hello, this is a preview of the designed voice.",
        language=req.language,
    )

    # Clone from the designed audio to get a reusable prompt
    clone_result = engine.clone_voice(design_audio)
    embedding_bytes = clone_result["prompt_bytes"]

    # Create profile in database
    profile_data = VoiceProfileCreate(
        name=req.name,
        description=req.description,
        language=req.language,
        tags=["designed"],
        channel_id=req.channel_id,
    )
    profile = profiles_module.create_profile(
        data=profile_data,
        engine_id=manager.active_model_id or "unknown",
        db=db,
    )

    # Add the designed audio as a sample
    profiles_module.add_sample(
        profile_id=profile.id,
        audio_bytes=design_audio,
        embedding_bytes=embedding_bytes,
        reference_text=req.description,
        duration_seconds=None,
        is_primary=True,
        db=db,
    )

    result = profiles_module._profile_to_dict(profile, sample_count=1)
    logger.info(f"Voice designed successfully: {profile.id}")
    return result


@app.post("/api/design-voice/preview")
async def preview_design_voice(req: PreviewDesignVoiceRequest):
    """Generate an ephemeral voice preview from a text description without saving."""
    logger.info("Generating voice design preview...")

    manager = get_manager()
    engine = manager.get_current_engine()

    design_audio = engine.design_voice(
        description=req.description,
        text=req.text,
        language=req.language,
    )

    return Response(content=design_audio, media_type="audio/wav")


# ==============================
# TTS Generation
# ==============================
@app.post("/api/generate")
async def generate_speech(req: GenerateRequest, db: Session = Depends(get_db)):
    """Generate speech audio from text using a saved voice."""
    logger.info(f"Generating speech: voice={req.voiceId}, lang={req.language}, emotion={req.emotion}")

    # Validate voice exists (check new DB first, then legacy)
    profile = profiles_module.get_profile(req.voiceId, db)
    if not profile:
        raise HTTPException(404, f"Voice not found: {req.voiceId}")

    embedding_path = profiles_module.get_profile_embedding_path(req.voiceId, db)
    if not embedding_path:
        raise HTTPException(404, f"Voice embedding not found: {req.voiceId}")

    # Optional parameters for advanced models
    kwargs = {}
    # Note: temperature/repetition_penalty are now managed by the engine
    # based on the selected emotion. Do NOT override them here.
    # Phase 2: Seed control for reproducibility
    if req.seed is not None:
        kwargs["seed"] = req.seed

    # Warn if voice was cloned with a different engine
    manager = get_manager()
    cloned_with = profile.engine_id or "unknown"
    if cloned_with != "unknown" and cloned_with != manager.active_model_id:
        logger.warning(
            f"Voice '{req.voiceId}' was cloned with '{cloned_with}' but current engine is '{manager.active_model_id}'. "
            f"Results may be degraded. For best quality, switch to the '{cloned_with}' model."
        )

    # ── Cache check ──
    cache_settings = {
        "language": req.language, "emotion": req.emotion,
        "speed": req.speed, "pitch": req.pitch,
        "style": req.style, "engine": manager.active_model_id,
        "seed": req.seed,
    }
    cached = get_cached(req.text, req.voiceId, **cache_settings)
    if cached:
        logger.info("Returning cached audio (cache HIT)")
        return Response(
            content=cached, media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav", "X-Cache": "HIT"},
        )

    engine = manager.get_current_engine()

    # ── Text chunking for long inputs ──
    chunks = chunk_text(req.text)

    if len(chunks) <= 1:
        # Short text — generate normally
        audio_bytes = engine.generate_speech(
            text=req.text,
            embedding_path=embedding_path,
            language=req.language,
            emotion=req.emotion,
            speed=req.speed,
            pitch=req.pitch,
            duration=req.duration,
            style=req.style,
            **kwargs
        )
    else:
        # Long text — generate each chunk and concatenate
        logger.info(f"Chunked generation: {len(chunks)} chunks")
        audio_segments = []
        for i, chunk_text_str in enumerate(chunks):
            logger.info(f"  Generating chunk {i+1}/{len(chunks)}: {chunk_text_str[:50]}...")
            chunk_audio = engine.generate_speech(
                text=chunk_text_str,
                embedding_path=embedding_path,
                language=req.language,
                emotion=req.emotion,
                speed=req.speed,
                pitch=req.pitch,
                style=req.style,
                **kwargs
            )
            # Decode WAV bytes to numpy array for concatenation
            chunk_data, chunk_sr = sf.read(io.BytesIO(chunk_audio))
            audio_segments.append(chunk_data)

        # Concatenate all segments with a small silence gap
        silence = np.zeros(int(chunk_sr * 0.15))  # 150ms gap between chunks
        combined = []
        for i, seg in enumerate(audio_segments):
            combined.append(seg)
            if i < len(audio_segments) - 1:
                combined.append(silence)
        combined_audio = np.concatenate(combined)

        buffer = io.BytesIO()
        sf.write(buffer, combined_audio, chunk_sr, format="WAV")
        buffer.seek(0)
        audio_bytes = buffer.getvalue()

    # Process audio through the mastering chain
    audio_bytes = master_audio(audio_bytes)

    # ── Cache store ──
    put_cached(audio_bytes, req.text, req.voiceId, **cache_settings)

    # ── Record in generation history ──
    try:
        history_module.record_generation(
            profile_id=req.voiceId,
            text=req.text,
            audio_bytes=audio_bytes,
            language=req.language,
            emotion=req.emotion,
            speed=req.speed,
            pitch=req.pitch,
            style=req.style,
            engine_id=manager.active_model_id or "unknown",
            db=db,
        )
    except Exception as e:
        logger.warning(f"Failed to record generation in history: {e}")

    logger.info(f"Speech generated and mastered: {len(audio_bytes)} bytes")
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav", "X-Cache": "MISS"},
    )


# ==============================
# Voice Preview
# ==============================
@app.post("/api/preview")
async def preview_voice(req: PreviewRequest, db: Session = Depends(get_db)):
    """Generate a short preview of a voice."""
    profile = profiles_module.get_profile(req.voiceId, db)
    if not profile:
        raise HTTPException(404, f"Voice not found: {req.voiceId}")

    embedding_path = profiles_module.get_profile_embedding_path(req.voiceId, db)
    if not embedding_path:
        raise HTTPException(404, f"Voice embedding not found: {req.voiceId}")

    manager = get_manager()
    engine = manager.get_current_engine()
    audio_bytes = engine.generate_speech(
        text=req.text,
        embedding_path=embedding_path,
        language=profile.language or "English",
    )

    return Response(content=audio_bytes, media_type="audio/wav")


# ==============================
# Voice Management
# ==============================
@app.get("/api/voices")
async def list_voices(db: Session = Depends(get_db)):
    """List all saved voices."""
    return profiles_module.list_profiles(db)


@app.get("/api/voices/{voice_id}")
async def get_voice_details(voice_id: str, db: Session = Depends(get_db)):
    """Get details of a specific voice."""
    profile = profiles_module.get_profile(voice_id, db)
    if not profile:
        raise HTTPException(404, f"Voice not found: {voice_id}")
    from sqlalchemy import func as sqlfunc
    from database import ProfileSample
    sample_count = db.query(sqlfunc.count(ProfileSample.id)).filter(ProfileSample.profile_id == voice_id).scalar()
    return profiles_module._profile_to_dict(profile, sample_count)


@app.delete("/api/voices/{voice_id}")
async def remove_voice(voice_id: str, db: Session = Depends(get_db)):
    """Delete a saved voice."""
    success = profiles_module.delete_profile(voice_id, db)
    if not success:
        raise HTTPException(404, f"Voice not found: {voice_id}")
    logger.info(f"Voice deleted: {voice_id}")
    return {"status": "deleted", "id": voice_id}


@app.get("/api/voices/{voice_id}/sample")
async def get_voice_sample(voice_id: str, db: Session = Depends(get_db)):
    """Get the original audio sample for a cloned voice."""
    sample_path = profiles_module.get_profile_sample_path(voice_id, db)
    if not sample_path:
        raise HTTPException(404, "No audio sample available for this voice")
    return FileResponse(sample_path, media_type="audio/wav")


# ============================================
# VOICE PROFILE SAMPLES (NEW — Multi-sample support)
# ============================================

@app.post("/api/voices/{voice_id}/samples")
async def add_voice_sample(
    voice_id: str,
    audio: UploadFile = File(...),
    reference_text: str = Form(""),
    db: Session = Depends(get_db),
):
    """Add an additional audio sample to a voice profile."""
    profile = profiles_module.get_profile(voice_id, db)
    if not profile:
        raise HTTPException(404, f"Voice not found: {voice_id}")

    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(400, "Empty audio file")

    # Validate reference audio
    is_valid, error_msg = validate_reference_audio(audio_bytes=audio_bytes)
    if not is_valid:
        logger.warning(f"Sample validation: {error_msg}")

    audio_bytes = sanitize_reference_audio(audio_bytes)

    # Clone to get embedding (with reference text for quality)
    manager = get_manager()
    engine = manager.get_current_engine()
    clone_result = engine.clone_voice(audio_bytes, ref_text=reference_text)
    embedding_bytes = clone_result["prompt_bytes"]
    
    # Use auto-transcribed text if engine returned it
    if not reference_text and "reference_text" in clone_result:
        reference_text = clone_result["reference_text"]

    duration = None
    try:
        data_arr, sr = sf.read(io.BytesIO(audio_bytes))
        duration = len(data_arr) / sr
    except Exception:
        pass

    sample = profiles_module.add_sample(
        profile_id=voice_id,
        audio_bytes=audio_bytes,
        embedding_bytes=embedding_bytes,
        reference_text=reference_text,
        duration_seconds=duration,
        is_primary=False,
        db=db,
    )
    return {"id": sample.id, "profile_id": voice_id, "duration": duration}


@app.get("/api/voices/{voice_id}/samples")
async def list_voice_samples(voice_id: str, db: Session = Depends(get_db)):
    """List all audio samples for a voice profile."""
    profile = profiles_module.get_profile(voice_id, db)
    if not profile:
        raise HTTPException(404, f"Voice not found: {voice_id}")
    samples = profiles_module.get_samples(voice_id, db)
    return [
        {
            "id": s.id,
            "profile_id": s.profile_id,
            "reference_text": s.reference_text,
            "duration_seconds": s.duration_seconds,
            "is_primary": s.is_primary,
            "audio_url": f"/api/voices/{voice_id}/samples/{s.id}/audio",
            "createdAt": s.created_at.isoformat() + "Z" if s.created_at else None,
        }
        for s in samples
    ]


@app.get("/api/voices/{voice_id}/samples/{sample_id}/audio")
async def get_sample_audio(voice_id: str, sample_id: str, db: Session = Depends(get_db)):
    """Get audio for a specific sample."""
    from database import ProfileSample
    sample = db.query(ProfileSample).filter(
        ProfileSample.id == sample_id,
        ProfileSample.profile_id == voice_id,
    ).first()
    if not sample or not sample.audio_path or not os.path.exists(sample.audio_path):
        raise HTTPException(404, "Sample audio not found")
    return FileResponse(sample.audio_path, media_type="audio/wav")


@app.delete("/api/voices/{voice_id}/samples/{sample_id}")
async def delete_voice_sample(voice_id: str, sample_id: str, db: Session = Depends(get_db)):
    """Delete a specific audio sample from a profile."""
    success = profiles_module.delete_sample(sample_id, db)
    if not success:
        raise HTTPException(404, "Sample not found")
    return {"status": "deleted", "id": sample_id}


# ============================================
# GENERATION HISTORY (NEW)
# ============================================

@app.get("/api/history")
async def get_history(
    profile_id: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """Get generation history with optional filtering."""
    return history_module.list_generations(
        db=db,
        profile_id=profile_id,
        search=search,
        limit=limit,
        offset=offset,
    )


@app.get("/api/history/{gen_id}/audio")
async def get_history_audio(gen_id: str, db: Session = Depends(get_db)):
    """Get audio for a historical generation."""
    gen = history_module.get_generation(gen_id, db)
    if not gen or not gen.audio_path or not os.path.exists(gen.audio_path):
        raise HTTPException(404, "Generation audio not found")
    return FileResponse(gen.audio_path, media_type="audio/wav")


@app.delete("/api/history/{gen_id}")
async def delete_history_item(gen_id: str, db: Session = Depends(get_db)):
    """Delete a generation from history."""
    success = history_module.delete_generation(gen_id, db)
    if not success:
        raise HTTPException(404, "Generation not found")
    return {"status": "deleted", "id": gen_id}


@app.delete("/api/history")
async def clear_all_history(
    profile_id: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Clear generation history."""
    count = history_module.clear_history(db, profile_id)
    return {"status": "cleared", "deleted_count": count}


# ==============================
# Streaming Generation (#2)
# ==============================
# StreamGenerateRequest imported from schemas.py


@app.post("/api/generate/stream")
async def generate_speech_stream(req: StreamGenerateRequest, db: Session = Depends(get_db)):
    """Stream audio chunks as they're generated via SSE (Server-Sent Events)."""
    profile = profiles_module.get_profile(req.voiceId, db)
    if not profile:
        raise HTTPException(404, f"Voice not found: {req.voiceId}")

    embedding_path = profiles_module.get_profile_embedding_path(req.voiceId, db)
    if not embedding_path:
        raise HTTPException(404, f"Voice embedding not found: {req.voiceId}")

    import base64

    def stream_chunks():
        manager = get_manager()
        engine = manager.get_current_engine()
        chunks = chunk_text(req.text)

        yield f"data: {json.dumps({'type': 'start', 'total_chunks': len(chunks)})}\n\n"

        for i, chunk_str in enumerate(chunks):
            try:
                chunk_audio = engine.generate_speech(
                    text=chunk_str,
                    embedding_path=embedding_path,
                    language=req.language,
                    emotion=req.emotion,
                    speed=req.speed,
                    pitch=req.pitch,
                    style=req.style,
                    seed=req.seed,
                )
                chunk_audio = master_audio(chunk_audio)

                yield f"data: {json.dumps({'type': 'chunk', 'index': i, 'total': len(chunks), 'audio_base64': base64.b64encode(chunk_audio).decode(), 'size_bytes': len(chunk_audio)})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'index': i, 'error': str(e)})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'total_chunks': len(chunks)})}\n\n"

    return StreamingResponse(stream_chunks(), media_type="text/event-stream")


# ==============================
# Async Job System (#1)
# ==============================
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


# AsyncGenerateRequest imported from schemas.py


def _run_async_job(job_id: str, req_data: dict):
    """Background thread that runs TTS generation and updates job status."""
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "processing"
            _jobs[job_id]["started_at"] = time.time()

        embedding_path = _get_embedding_path_compat(req_data["voiceId"])
        if not embedding_path:
            raise RuntimeError(f"Voice embedding not found: {req_data['voiceId']}")

        manager = get_manager()
        engine = manager.get_current_engine()

        chunks = chunk_text(req_data["text"])
        if len(chunks) <= 1:
            audio_bytes = engine.generate_speech(
                text=req_data["text"],
                embedding_path=embedding_path,
                language=req_data.get("language", "English"),
                emotion=req_data.get("emotion", "neutral"),
                speed=req_data.get("speed", 1.0),
                pitch=req_data.get("pitch", 1.0),
                style=req_data.get("style"),
                seed=req_data.get("seed"),
            )
        else:
            audio_segments = []
            for i, chunk_str in enumerate(chunks):
                with _jobs_lock:
                    _jobs[job_id]["progress"] = round((i / len(chunks)) * 100)
                    _jobs[job_id]["message"] = f"Generating chunk {i+1}/{len(chunks)}"
                chunk_audio = engine.generate_speech(
                    text=chunk_str,
                    embedding_path=embedding_path,
                    language=req_data.get("language", "English"),
                    emotion=req_data.get("emotion", "neutral"),
                    speed=req_data.get("speed", 1.0),
                    pitch=req_data.get("pitch", 1.0),
                    style=req_data.get("style"),
                    seed=req_data.get("seed"),
                )
                chunk_data, chunk_sr = sf.read(io.BytesIO(chunk_audio))
                audio_segments.append(chunk_data)

            silence = np.zeros(int(chunk_sr * 0.15))
            combined = []
            for i, seg in enumerate(audio_segments):
                combined.append(seg)
                if i < len(audio_segments) - 1:
                    combined.append(silence)
            combined_audio = np.concatenate(combined)
            buf = io.BytesIO()
            sf.write(buf, combined_audio, chunk_sr, format="WAV")
            buf.seek(0)
            audio_bytes = buf.getvalue()

        audio_bytes = master_audio(audio_bytes)

        result_path = os.path.join(os.path.dirname(__file__), "data", "jobs", f"{job_id}.wav")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "wb") as f:
            f.write(audio_bytes)

        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["progress"] = 100
            _jobs[job_id]["message"] = "Done"
            _jobs[job_id]["result_size"] = len(audio_bytes)
            _jobs[job_id]["completed_at"] = time.time()
            _jobs[job_id]["duration"] = round(time.time() - _jobs[job_id]["started_at"], 2)

    except Exception as e:
        logger.error(f"Async job {job_id} failed: {e}", exc_info=True)
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)


@app.post("/api/generate/async")
async def generate_speech_async(req: AsyncGenerateRequest):
    """Submit a TTS job for async processing. Returns a job ID to poll."""
    voice = _get_voice_compat(req.voiceId)
    if not voice:
        raise HTTPException(404, f"Voice not found: {req.voiceId}")

    job_id = str(uuid.uuid4())
    req_data = req.model_dump()

    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Queued for processing",
            "created_at": time.time(),
            "voice_id": req.voiceId,
            "text_preview": req.text[:100],
        }

    thread = threading.Thread(target=_run_async_job, args=(job_id, req_data), daemon=True)
    thread.start()

    logger.info(f"Async job submitted: {job_id}")
    return {"jobId": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Poll for the status of an async generation job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    return job


@app.get("/api/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """Download the result of a completed async job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    if job["status"] != "completed":
        raise HTTPException(400, f"Job is not complete yet. Status: {job['status']}")

    result_path = os.path.join(os.path.dirname(__file__), "data", "jobs", f"{job_id}.wav")
    if not os.path.exists(result_path):
        raise HTTPException(404, "Job result file not found")
    return FileResponse(result_path, media_type="audio/wav",
                        headers={"Content-Disposition": f"attachment; filename={job_id}.wav"})


# BatchItem and BatchGenerateRequest imported from schemas.py


@app.post("/api/generate/batch")
async def generate_batch(req: BatchGenerateRequest, db: Session = Depends(get_db)):
    """Generate multiple texts in a single request."""
    profile = profiles_module.get_profile(req.voiceId, db)
    if not profile:
        raise HTTPException(404, f"Voice not found: {req.voiceId}")

    embedding_path = profiles_module.get_profile_embedding_path(req.voiceId, db)
    if not embedding_path:
        raise HTTPException(404, f"Voice embedding not found: {req.voiceId}")

    if len(req.items) > 20:
        raise HTTPException(400, "Maximum 20 items per batch")

    manager = get_manager()
    engine = manager.get_current_engine()

    import base64
    results = []
    for i, item in enumerate(req.items):
        logger.info(f"Batch item {i+1}/{len(req.items)}: {item.text[:50]}...")
        try:
            audio_bytes = engine.generate_speech(
                text=item.text,
                embedding_path=embedding_path,
                language=item.language,
                emotion=item.emotion,
                speed=item.speed,
            )
            audio_bytes = master_audio(audio_bytes)
            results.append({
                "index": i,
                "status": "success",
                "size_bytes": len(audio_bytes),
                "audio_base64": base64.b64encode(audio_bytes).decode(),
            })
        except Exception as e:
            results.append({"index": i, "status": "failed", "error": str(e)})

    return {"voiceId": req.voiceId, "total": len(req.items), "results": results}


# ==============================
# Health Check (#9)
# ==============================
@app.get("/api/health")
async def health_check():
    """Basic API health check."""
    return {"status": "ok", "version": "2.0.0"}


@app.get("/api/health/engine")
async def engine_health_check():
    """Check if the loaded engine is healthy and can generate."""
    manager = get_manager()

    if not manager.active_model_id:
        return {"status": "no_model", "message": "No model is currently loaded.", "model_id": None}

    engine = manager.loaded_engines.get(manager.active_model_id)
    if not engine or not engine.is_loaded:
        return {
            "status": "not_loaded",
            "message": f"Model {manager.active_model_id} is registered but not loaded.",
            "model_id": manager.active_model_id,
        }

    model_info = manager.AVAILABLE_MODELS.get(manager.active_model_id, {})
    return {
        "status": "ready",
        "model_id": manager.active_model_id,
        "model_name": model_info.get("name", "Unknown"),
        "capabilities": engine.get_capabilities(),
        "features": model_info.get("features", []),
        "vram_estimate": model_info.get("vram_estimate", "Unknown"),
    }


# ==============================
# VRAM Monitoring (#10)
# ==============================
@app.get("/api/system/vram")
async def get_vram_info():
    """Get GPU memory usage information."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"gpu_available": False, "message": "No CUDA GPU detected. Running on CPU."}

        device_count = torch.cuda.device_count()
        gpus = []
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            total = props.total_mem
            gpus.append({
                "index": i,
                "name": props.name,
                "total_mb": round(total / (1024 * 1024)),
                "allocated_mb": round(allocated / (1024 * 1024)),
                "free_mb": round((total - allocated) / (1024 * 1024)),
                "usage_percent": round(allocated / total * 100, 1),
            })

        manager = get_manager()
        return {
            "gpu_available": True,
            "device_count": device_count,
            "gpus": gpus,
            "active_model": manager.active_model_id,
            "loaded_models": list(manager.loaded_engines.keys()),
        }
    except ImportError:
        return {"gpu_available": False, "message": "PyTorch not installed."}


# ==============================
# Cache Management (#5)
# ==============================
@app.get("/api/cache/stats")
async def cache_stats():
    """Get audio cache statistics."""
    return get_cache_stats()


@app.delete("/api/cache")
async def api_clear_cache():
    """Clear all cached audio."""
    count = clear_audio_cache()
    return {"status": "cleared", "files_removed": count}


# ==============================
# Graceful Error Handler (#8)
# ==============================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all that returns friendly error messages instead of generic 500s."""
    error_msg = str(exc)

    if "out of memory" in error_msg.lower() or "OutOfMemoryError" in error_msg:
        logger.error("CUDA Out of Memory caught. Cleaning up VRAM...")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except:
            pass
        return JSONResponse(status_code=507, content={
            "detail": "GPU out of memory. Try a smaller model, shorter text, or restart the API.",
            "error": "GPU memory full",
            "message": "Not enough GPU memory. Try a smaller model, shorter text, or restart the API.",
            "code": "OOM",
        })
    elif "failed to load" in error_msg.lower():
        return JSONResponse(status_code=503, content={
            "detail": f"Failed to load the AI model: {error_msg}",
            "error": "Model load failure",
            "message": f"Failed to load the AI model: {error_msg}",
            "code": "MODEL_LOAD_FAILED",
        })
    elif "Reference audio not found" in error_msg:
        return JSONResponse(status_code=404, content={
            "detail": f"{error_msg}. Try re-cloning the voice.",
            "error": "Missing reference audio",
            "message": f"{error_msg}. Try re-cloning the voice.",
            "code": "REF_AUDIO_MISSING",
        })
    else:
        logger.error(f"Unhandled error: {exc}", exc_info=True)
        return JSONResponse(status_code=500, content={
            "detail": f"An unexpected error occurred: {error_msg}",
            "error": "Internal server error",
            "message": f"An unexpected error occurred: {error_msg}",
            "code": "INTERNAL",
        })


# ================================================================
# PART 3: NEW FEATURES
# ================================================================


# ==============================
# #3 — Multi-Speaker Conversation
# ==============================
# ConversationRequest imported from schemas.py


@app.post("/api/conversation")
async def generate_conversation(req: ConversationRequest, db: Session = Depends(get_db)):
    """Generate a multi-speaker conversation from a labeled script."""
    logger.info(f"Multi-speaker conversation: {len(req.voices)} speakers")

    # Validate all voices exist and build embedding map
    voice_map = {}
    for label, voice_id in req.voices.items():
        embedding = profiles_module.get_profile_embedding_path(voice_id, db)
        if not embedding:
            raise HTTPException(404, f"Voice not found for speaker '{label}': {voice_id}")
        voice_map[label] = embedding

    script_lines = parse_multi_speaker_script(req.script)
    if not script_lines:
        raise HTTPException(400, "Could not parse any speaker lines from the script")

    # Check that all speakers in script have voices assigned
    speakers_in_script = set(line["speaker"] for line in script_lines)
    missing = speakers_in_script - set(req.voices.keys())
    if missing:
        raise HTTPException(400, f"No voice assigned for speaker(s): {', '.join(missing)}")

    manager = get_manager()
    engine = manager.get_current_engine()

    audio_bytes = generate_multi_speaker_audio(
        script_lines=script_lines,
        voice_map=voice_map,
        engine=engine,
        default_language=req.language,
        gap_seconds=req.gap,
    )
    audio_bytes = master_audio(audio_bytes)

    return Response(content=audio_bytes, media_type="audio/wav",
                    headers={"Content-Disposition": "attachment; filename=conversation.wav"})


# ==============================
# #4 — Audiobook Generator
# ==============================
# AudiobookRequest imported from schemas.py


@app.post("/api/audiobook")
async def generate_audiobook(req: AudiobookRequest, db: Session = Depends(get_db)):
    """Generate a full audiobook from text with chapter detection and dialogue voices."""
    logger.info(f"Audiobook generation: '{req.title}'")

    narrator_embedding = profiles_module.get_profile_embedding_path(req.narratorVoiceId, db)
    if not narrator_embedding:
        raise HTTPException(404, f"Narrator voice not found: {req.narratorVoiceId}")

    dialogue_embedding = narrator_embedding
    if req.dialogueVoiceId:
        dialogue_embedding = profiles_module.get_profile_embedding_path(req.dialogueVoiceId, db) or narrator_embedding

    manager = get_manager()
    engine = manager.get_current_engine()

    chapters = split_into_chapters(req.text)
    logger.info(f"Detected {len(chapters)} chapters")

    all_segments = []
    sample_rate = None
    chapter_markers = []

    for ch_idx, chapter in enumerate(chapters):
        logger.info(f"  Chapter {ch_idx+1}/{len(chapters)}: {chapter['title']}")
        chapter_start_samples = sum(len(s) for s in all_segments)

        # Detect dialogue within this chapter
        parts = detect_dialogue(chapter["content"])

        for part in parts:
            emb = dialogue_embedding if part["type"] == "dialogue" else narrator_embedding
            audio_bytes = engine.generate_speech(
                text=part["text"],
                embedding_path=emb,
                language=req.language,
            )
            data, sr = sf.read(io.BytesIO(audio_bytes))
            if sample_rate is None:
                sample_rate = sr
            all_segments.append(data)

            # Add small gap between parts
            all_segments.append(np.zeros(int(sr * 0.2)))

        # Add longer gap between chapters
        all_segments.append(np.zeros(int(sample_rate * 1.0)))

        chapter_markers.append({
            "chapter": ch_idx + 1,
            "title": chapter["title"],
            "start_sample": chapter_start_samples,
        })

    # Concatenate
    combined = np.concatenate(all_segments)
    buffer = io.BytesIO()
    sf.write(buffer, combined, sample_rate, format="WAV")
    buffer.seek(0)
    audio_bytes = master_audio(buffer.getvalue())

    logger.info(f"Audiobook generated: {len(audio_bytes)} bytes, {len(chapters)} chapters")
    return Response(content=audio_bytes, media_type="audio/wav",
                    headers={"Content-Disposition": f"attachment; filename={req.title}.wav"})


# ==============================
# #5 — Voice Library (Export/Import)
# ==============================
VOICES_DIR = os.path.join(os.path.dirname(__file__), "data", "voices")


@app.get("/api/voices/{voice_id}/export")
async def export_voice_endpoint(voice_id: str, db: Session = Depends(get_db)):
    """Export a voice as a downloadable .resound file."""
    profile = profiles_module.get_profile(voice_id, db)
    if not profile:
        raise HTTPException(404, f"Voice not found: {voice_id}")

    try:
        resound_bytes = export_voice(voice_id, db)
        return Response(
            content=resound_bytes,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={voice_id}.resound"},
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.post("/api/voices/import")
async def import_voice_endpoint(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Import a .resound file to add a voice."""
    if not file.filename.endswith(".resound"):
        raise HTTPException(400, "File must have .resound extension")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(400, "Empty file")

    try:
        meta = import_voice(file_bytes, db, VOICES_DIR)
        logger.info(f"Voice imported: {meta['id']} (name: {meta.get('name', 'unknown')})")
        return JSONResponse(status_code=201, content=meta)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/transcribe")
async def transcribe_audio_endpoint(audio: UploadFile = File(...)):
    """Transcribe audio bytes using Whisper for better cloning accuracy."""
    try:
        content = await audio.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            import whisper
            # Use 'base' model for speed in transcription
            model = whisper.load_model("base")
            result = model.transcribe(tmp_path)
            return {"text": result.get("text", "").strip()}
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(500, detail=f"Transcription failed: {str(e)}")


# ==============================
# #6 — Subtitle/SRT Generator
# ==============================
# SrtRequest imported from schemas.py


@app.post("/api/generate/srt")
async def generate_with_srt(req: SrtRequest, db: Session = Depends(get_db)):
    """Generate audio AND matching SRT subtitle file."""
    profile = profiles_module.get_profile(req.voiceId, db)
    if not profile:
        raise HTTPException(404, f"Voice not found: {req.voiceId}")

    embedding_path = profiles_module.get_profile_embedding_path(req.voiceId, db)
    if not embedding_path:
        raise HTTPException(404, f"Voice embedding not found: {req.voiceId}")

    manager = get_manager()
    engine = manager.get_current_engine()

    # Generate audio
    audio_bytes = engine.generate_speech(
        text=req.text, embedding_path=embedding_path, language=req.language,
    )
    audio_bytes = master_audio(audio_bytes)

    # Calculate duration
    audio_data, sr = sf.read(io.BytesIO(audio_bytes))
    total_duration = len(audio_data) / sr

    # Split text into chunks for subtitle segments
    from utils.text_chunker import split_into_sentences
    sentences = split_into_sentences(req.text)
    segments = estimate_segment_timing(sentences, total_duration)
    srt_content = generate_srt(segments)

    import base64
    return {
        "audio_base64": base64.b64encode(audio_bytes).decode(),
        "audio_size": len(audio_bytes),
        "duration_seconds": round(total_duration, 2),
        "srt": srt_content,
        "segments": segments,
    }


# ==============================
# #7 — Background Music Mixing
# ==============================
@app.post("/api/mix-music")
async def mix_music(
    voice_audio: UploadFile = File(...),
    music_audio: UploadFile = File(...),
    music_volume: float = Form(0.15),
    fade_in: float = Form(1.0),
    fade_out: float = Form(2.0),
):
    """Mix background music with voice audio."""
    voice_bytes = await voice_audio.read()
    music_bytes = await music_audio.read()

    if not voice_bytes or not music_bytes:
        raise HTTPException(400, "Both voice and music audio files are required")

    mixed = mix_audio_with_music(
        voice_bytes=voice_bytes,
        music_bytes=music_bytes,
        music_volume=music_volume,
        fade_in_seconds=fade_in,
        fade_out_seconds=fade_out,
    )

    return Response(content=mixed, media_type="audio/wav",
                    headers={"Content-Disposition": "attachment; filename=mixed.wav"})


# ==============================
# #8 — Emotion Timeline
# ==============================
# EmotionSegment and EmotionTimelineRequest imported from schemas.py


@app.post("/api/generate/emotion-timeline")
async def generate_emotion_timeline(req: EmotionTimelineRequest, db: Session = Depends(get_db)):
    """Generate speech with different emotions per sentence."""
    profile = profiles_module.get_profile(req.voiceId, db)
    if not profile:
        raise HTTPException(404, f"Voice not found: {req.voiceId}")

    embedding_path = profiles_module.get_profile_embedding_path(req.voiceId, db)
    if not embedding_path:
        raise HTTPException(404, f"Voice embedding not found: {req.voiceId}")

    timeline = parse_emotion_timeline([s.model_dump() for s in req.segments])

    manager = get_manager()
    engine = manager.get_current_engine()

    audio_parts = []
    sample_rate = None
    for i, seg in enumerate(timeline):
        logger.info(f"  Timeline segment {i+1}/{len(timeline)}: [{seg['emotion']}] {seg['text'][:40]}...")
        audio_bytes = engine.generate_speech(
            text=seg["text"],
            embedding_path=embedding_path,
            language=req.language,
            emotion=seg["emotion"],
            speed=req.speed,
        )
        data, sr = sf.read(io.BytesIO(audio_bytes))
        if sample_rate is None:
            sample_rate = sr
        audio_parts.append(data)

    # Concatenate with tiny gaps
    silence = np.zeros(int(sample_rate * 0.1))
    combined = []
    for i, part in enumerate(audio_parts):
        combined.append(part)
        if i < len(audio_parts) - 1:
            combined.append(silence)

    result = np.concatenate(combined)
    buffer = io.BytesIO()
    sf.write(buffer, result, sample_rate, format="WAV")
    buffer.seek(0)
    audio_bytes = master_audio(buffer.getvalue())

    return Response(content=audio_bytes, media_type="audio/wav",
                    headers={"Content-Disposition": "attachment; filename=emotion_timeline.wav"})


# ==============================
# Whisper Auto-Transcription (Turbo model)
# ==============================
@app.post("/api/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
):
    """
    Transcribe audio using Whisper Turbo (large-v3-turbo).
    Auto-detects language and returns the transcription text.
    Used by the frontend to pre-fill reference text for voice cloning.
    """
    logger.info(f"Transcribing audio: {audio.filename}")
    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(400, "Empty audio file")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        try:
            import whisper
            model = whisper.load_model("turbo")
            result = model.transcribe(tmp_path, language=None)
            text = result.get("text", "").strip()
            detected_language = result.get("language", "en")
            
            return {
                "text": text,
                "language": detected_language,
                "success": True,
            }
        except ImportError:
            raise HTTPException(
                501,
                "Whisper is not installed. Install with: pip install openai-whisper"
            )
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            raise HTTPException(500, f"Transcription failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


# ==============================
# Podcast Studio
# ==============================
@app.post("/api/podcast")
async def generate_podcast(req: PodcastTimelineRequest, db: Session = Depends(get_db)):
    """Generate a multi-speaker podcast. Each block has a voice_id and text."""
    logger.info(f"Generating podcast: {req.story_name} ({len(req.blocks)} segments)")

    if not req.blocks:
        raise HTTPException(400, "No podcast blocks provided.")

    manager = get_manager()
    engine = manager.get_current_engine()

    segments = []
    sample_rate = None

    for i, block in enumerate(req.blocks):
        profile = profiles_module.get_profile(block.voice_id, db)
        if not profile:
            raise HTTPException(404, f"Voice not found: {block.voice_id}")

        embedding = profiles_module.get_profile_embedding_path(block.voice_id, db)
        if not embedding:
            raise HTTPException(404, f"Voice embedding not found: {block.voice_id}")

        logger.info(f"  Podcast [{i+1}/{len(req.blocks)}]: {block.text[:50]}...")

        try:
            audio_bytes = engine.generate_speech(
                text=block.text,
                embedding_path=embedding,
                language=req.language,
            )
            data, sr = sf.read(io.BytesIO(audio_bytes))
            if sample_rate is None:
                sample_rate = sr
            segments.append(data)
        except Exception as e:
            logger.error(f"  Failed segment {i+1}: {e}")
            if sample_rate:
                segments.append(np.zeros(int(sample_rate * 0.5)))
            continue

    if not segments or sample_rate is None:
        raise HTTPException(500, "Failed to generate any audio segments")

    # Concatenate with pauses between speakers
    pause = np.zeros(int(sample_rate * 0.4), dtype=np.float32)
    combined = []
    for i, seg in enumerate(segments):
        combined.append(seg.astype(np.float32))
        if i < len(segments) - 1:
            combined.append(pause)

    result = np.concatenate(combined)
    if np.abs(result).max() > 0:
        result = result / np.abs(result).max() * 0.95

    buffer = io.BytesIO()
    sf.write(buffer, result, sample_rate, format="WAV")
    buffer.seek(0)
    audio_bytes = master_audio(buffer.getvalue())

    logger.info(f"Podcast generated: {len(req.blocks)} segments, {len(audio_bytes)} bytes")
    return Response(
        content=audio_bytes, media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=podcast.wav"},
    )


# ==============================
# GPU Stats & Model Management
# ==============================
@app.get("/api/models/gpu-stats")
async def get_gpu_stats():
    """Get real-time GPU memory and utilization stats."""
    gpus = []
    error = None
    
    # Try pynvml first for detailed stats
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            gpus.append({
                "index": i,
                "name": name,
                "memory_used_mb": round(mem.used / 1048576),
                "memory_total_mb": round(mem.total / 1048576),
                "memory_free_mb": round(mem.free / 1048576),
                "gpu_util_percent": util.gpu,
                "memory_util_percent": util.memory,
                "driver": "pynvml"
            })
    except Exception as nvml_err:
        error = f"NVML Error: {nvml_err}"
        logger.warning(f"pynvml failed: {nvml_err}")

    # Fallback to torch for basic memory info if pynvml failed or is partial
    if not gpus:
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    total_mem = torch.cuda.get_device_properties(i).total_memory
                    used_mem = torch.cuda.memory_reserved(i) # approx
                    gpus.append({
                        "index": i,
                        "name": f"{name} (via Torch)",
                        "memory_used_mb": round(used_mem / 1048576),
                        "memory_total_mb": round(total_mem / 1048576),
                        "memory_free_mb": round((total_mem - used_mem) / 1048576),
                        "gpu_util_percent": 0, # torch cant get util easily
                        "memory_util_percent": 0,
                        "driver": "torch"
                    })
        except Exception as torch_err:
            error = f"{error} | Torch Error: {torch_err}" if error else str(torch_err)

    return {"gpus": gpus, "error": error}


@app.post("/api/models/unload-all")
async def unload_all_models():
    """Unload all models from GPU VRAM."""
    manager = get_manager()
    unloaded = []
    for model_id in list(manager.loaded_engines.keys()):
        try:
            eng = manager.loaded_engines[model_id]
            if hasattr(eng, "unload"):
                eng.unload()
            del manager.loaded_engines[model_id]
            unloaded.append(model_id)
        except Exception as e:
            logger.warning(f"Failed to unload {model_id}: {e}")
    manager.active_model_id = None

    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Unloaded all models: {unloaded}")
    return {"status": "success", "unloaded": unloaded}


@app.post("/api/models/{model_id}/unload")
async def unload_specific_model(model_id: str):
    """Unload a specific model from GPU VRAM."""
    manager = get_manager()
    if model_id not in manager.loaded_engines:
        raise HTTPException(404, f"Model not loaded: {model_id}")

    try:
        eng = manager.loaded_engines[model_id]
        if hasattr(eng, "unload"):
            eng.unload()
        del manager.loaded_engines[model_id]
    except Exception as e:
        raise HTTPException(500, f"Failed to unload: {e}")

    if manager.active_model_id == model_id:
        manager.active_model_id = None

    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Unloaded model: {model_id}")
    return {"status": "success", "model_id": model_id}


# ==============================
# Real-Time Download Progress (SSE)
# ==============================
@app.get("/api/models/progress/{model_name}")
async def get_model_progress(model_name: str):
    """
    SSE endpoint for real-time model download/load progress.
    Sends progress events with: progress %, filename, speed, ETA.
    Sends heartbeat every 1s to keep connection alive.
    """
    progress_manager = get_progress_manager()
    
    return StreamingResponse(
        progress_manager.subscribe(model_name),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ==============================
# Hardware Accelerator Detection
# ==============================
@app.get("/api/accelerators")
async def get_accelerators():
    """
    Detect available hardware accelerators.
    Returns info about CUDA, MPS, DirectML, XPU, MLX support.
    """
    return detect_accelerators()


# ==============================
# Profile Import/Export (.resound.zip)
# ==============================
@app.get("/api/voices/{voice_id}/export")
async def export_voice_profile(voice_id: str, db: Session = Depends(get_db)):
    """Export a voice profile as a .resound.zip file."""
    zip_bytes = profiles_module.export_profile(voice_id, db)
    if not zip_bytes:
        raise HTTPException(404, f"Voice not found: {voice_id}")
    
    profile = profiles_module.get_profile(voice_id, db)
    safe_name = (profile.name or "voice").replace(" ", "_").replace("/", "_")
    
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={safe_name}.resound.zip",
        },
    )


@app.post("/api/profiles/import")
async def import_voice_profile(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Import a voice profile from a .resound.zip file."""
    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(400, "Please upload a .resound.zip file")
    
    zip_bytes = await file.read()
    if len(zip_bytes) == 0:
        raise HTTPException(400, "Empty file")

    manager = get_manager()
    profile = profiles_module.import_profile(
        zip_bytes=zip_bytes,
        engine_id=manager.active_model_id or "unknown",
        db=db,
    )
    
    if not profile:
        raise HTTPException(500, "Failed to import voice profile")
    
    from sqlalchemy import func as sqlfunc
    from database import ProfileSample
    sample_count = db.query(sqlfunc.count(ProfileSample.id)).filter(
        ProfileSample.profile_id == profile.id
    ).scalar()
    
    result = profiles_module._profile_to_dict(profile, sample_count)
    logger.info(f"Voice profile imported: {profile.id} ({profile.name})")
    return result


# ==============================
# Streaming WAV Generation (64KB chunks)
# ==============================
@app.post("/api/generate/stream-wav")
async def generate_speech_stream_wav(req: StreamGenerateRequest, db: Session = Depends(get_db)):
    """
    Stream generated audio as raw WAV in 64KB chunks.
    Unlike the SSE stream endpoint, this returns a single WAV file
    streamed incrementally for better playback experience.
    """
    profile = profiles_module.get_profile(req.voiceId, db)
    if not profile:
        raise HTTPException(404, f"Voice not found: {req.voiceId}")

    embedding_path = profiles_module.get_profile_embedding_path(req.voiceId, db)
    if not embedding_path:
        raise HTTPException(404, f"Voice embedding not found: {req.voiceId}")

    manager = get_manager()
    engine = manager.get_current_engine()
    
    kwargs = {}
    if req.seed is not None:
        kwargs["seed"] = req.seed

    # Generate full audio first (true streaming would require model-level support)
    audio_bytes = engine.generate_speech(
        text=req.text,
        embedding_path=embedding_path,
        language=req.language,
        emotion=req.emotion,
        speed=req.speed,
        pitch=req.pitch,
        style=req.style,
        **kwargs,
    )
    audio_bytes = master_audio(audio_bytes)

    async def _wav_stream():
        """Yield WAV audio in 64KB chunks."""
        chunk_size = 64 * 1024  # 64KB
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i:i + chunk_size]

    return StreamingResponse(
        _wav_stream(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=output.wav",
            "Content-Length": str(len(audio_bytes)),
        },
    )

