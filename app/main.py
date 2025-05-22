from fastapi import FastAPI, HTTPException
from app.models import TranslationRequest, TranslationResponse
from app.translation import translate, detect_language

app = FastAPI()

@app.post("/translate", response_model=TranslationResponse)
async def translate_text(req: TranslationRequest):
    try:
        src_lang = req.src_lang or detect_language(req.text)
        translated = await run_blocking_translate(req.text, src_lang, req.tgt_lang)
        return TranslationResponse(translation=translated)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Async wrapper
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor()

async def run_blocking_translate(*args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, translate, *args)