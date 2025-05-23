from fastapi import FastAPI, HTTPException, Request
from app.models import TranslationRequest, TranslationResponse
from app.translation import translate, detect_language, get_best_browser_lang
app = FastAPI()

# TO DO: Check best practices for request error handling
# TO DO: Implement response caching, i.e. don't translate the same thing twice
@app.post("/translate", response_model=TranslationResponse)
async def translate_text(req: TranslationRequest, request: Request):
    try:
        # Auto-detect source language if missing
        src_lang = req.src_lang or detect_language(req.text)

        # Auto-detect target language if missing
        tgt_lang = req.tgt_lang
        if not tgt_lang:
            accept_langs = request.headers.get("accept-language", "")
            tgt_lang = get_best_browser_lang(accept_langs)

        # Skip translation if same language
        if tgt_lang == src_lang:
            return TranslationResponse(
                translation=req.text,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                translated=False
            )

        translated = await run_blocking_translate(req.text, src_lang, tgt_lang)
        return TranslationResponse(
            translation=translated,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            translated=True
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Async wrapper
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor()

async def run_blocking_translate(*args, use_llm=True, **kwargs):
    return await translate(*args, use_llm=use_llm, **kwargs)