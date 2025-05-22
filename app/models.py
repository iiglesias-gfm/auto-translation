from pydantic import BaseModel
from typing import Optional

class TranslationRequest(BaseModel):
    text: str
    src_lang: Optional[str]  # ISO 639-1
    tgt_lang: str
    campaign_id: str

class TranslationResponse(BaseModel):
    translation: str
