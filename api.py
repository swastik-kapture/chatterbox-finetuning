from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from chatterbox.tts import ChatterboxTTS
from scipy.io.wavfile import write
import io
import re
import gc
import torch
from pathlib import Path

app = FastAPI(
    title="ChatterBox TTS",
    summary="Generate speech using ChatterBox TTS.",
    version="0.1",
    redoc_url=f"/v1/chatterbox/redoc",
    docs_url=f"/v1/chatterbox/docs",
    openapi_url=f"/v1/chatterbox/openapi.json",
    swagger_ui_parameters={"displayRequestDuration": True, "displayOperationId": True},
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
templates = Jinja2Templates(directory="templates")

available_voices = ["default"]
model = ChatterboxTTS.from_pretrained(device="cuda")
model.prepare_conditionals("aysha_0002.wav")
sample_rate = 24000


class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str = "aysha_0002"
    response_format: str = "wav"
    stream: bool = False
    audio_prompt_path: str = ""


def audio_chunk_to_wav_bytes(audio_chunk: torch.Tensor) -> bytes:
    audio_np = audio_chunk.squeeze().cpu().numpy()
    audio_int16 = (audio_np * 32767).astype("int16")
    buffer = io.BytesIO()
    write(buffer, sample_rate, audio_int16)
    return buffer.getvalue()


def audio_chunk_to_pcm_bytes(audio_chunk: torch.Tensor) -> bytes:
    audio_np = audio_chunk.squeeze().cpu().numpy()
    audio_int16 = (audio_np * 32767).astype("int16")
    return audio_int16.tobytes()


def split_into_sentences(text: str) -> list[str]:
    sentence_endings = re.compile(r"(?<=[.!?]) +")
    return sentence_endings.split(text.strip())


@app.post("/v1/chatterbox/speech")
async def generate_speech(req: SpeechRequest):
    text = req.input
    stream = req.stream

    if req.response_format != "wav":
        return Response(status_code=400, content="Only 'wav' format is supported.")

    if req.audio_prompt_path:
        candidate_path = UPLOAD_DIR / req.audio_prompt_path
        if not candidate_path.is_file():
            raise HTTPException(
                status_code=404, detail="Specified audio_prompt_path not found."
            )
        model.prepare_conditionals(str(candidate_path))

    if stream:

        async def audio_stream():
            try:
                sentence_count = 0
                # for sentence in split_into_sentences(text):
                #     if not sentence.strip():
                #         continue
                #     sentence_count += 1
                #     print(f"Synthesizing sentence: {sentence}")
                for chunk, _ in model.generate_stream(
                    text,
                    exaggeration=0.5,
                    cfg_weight=0.5,
                    temperature=0.3,
                    chunk_size=25,
                ):
                    wav_bytes = audio_chunk_to_pcm_bytes(chunk)
                    yield wav_bytes
            finally:
                gc.collect()
                torch.cuda.empty_cache()

        return StreamingResponse(
            audio_stream(),
            media_type="application/octet-stream",
            headers={
                "Content-Type": "audio/wav",
                "Transfer-Encoding": "chunked",
            },
        )
    else:
        try:
            output_waveform: torch.Tensor = model.generate(
                text=text,
                audio_prompt_path=str(candidate_path) if req.audio_prompt_path else None,
                exaggeration=0.5,
                cfg_weight=0.5,
                temperature=0.5,
            )
            wav_bytes = audio_chunk_to_wav_bytes(output_waveform)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Type": "audio/wav",
                "Content-Disposition": "attachment; filename=speech.wav",
            },
        )


@app.post("/v1/chatterbox/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    """
    Accept an uploaded .wav (or any audio) file, save it under 'uploads/'
    and return the filename. The frontend will then list it under available prompts.
    """
    filename = Path(file.filename).name  # sanitize
    dest = UPLOAD_DIR / filename

    with open(dest, "wb") as out_file:
        contents = await file.read()
        out_file.write(contents)

    return JSONResponse({"filename": filename})


@app.get("/v1/chatterbox/list_audio")
async def list_audio_files():
    files = [f.name for f in UPLOAD_DIR.iterdir() if f.is_file()]
    return JSONResponse({"files": files})


@app.get("/v1/chatterbox/ui", response_class=HTMLResponse)
async def frontend(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "voices": available_voices}
    )