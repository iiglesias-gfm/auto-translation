from pydantic import BaseModel

class TranslationRequest(BaseModel):
    text: str
    src_lang: str  # ISO 639-1
    tgt_lang: str
    campaign_id: str

class TranslationResponse(BaseModel):
    translation: str
