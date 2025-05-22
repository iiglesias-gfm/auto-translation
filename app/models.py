from pydantic import BaseModel
from typing import Optional

class TranslationRequest(BaseModel):
    text: str = "Help me help people help each other"
    src_lang: Optional[str] = "en" # ISO 639-1
    tgt_lang: Optional[str] = "es"
    campaign_id: str = "1234"

class TranslationResponse(BaseModel):
    translation: str = "Ayúdame a ayudar a la gente a ayudarse unos a otros."
    src_lang: str = "en"
    tgt_lang: str = "es"
    translated: bool = True
