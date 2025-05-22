from transformers import (
    MarianMTModel, MarianTokenizer,
    M2M100ForConditionalGeneration, M2M100Tokenizer
)
import torch
import re
import unicodedata
import nltk
nltk.download("punkt")
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from langdetect import detect

model_cache = {}
M2M_LANGUAGES = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M").lang_code_to_id.keys()

# Preprocessing: normalize and mask entities
def preprocess_text(text: str):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    text, replacements = mask_entities(text)
    return text, replacements

def mask_entities(text: str):
    url_pattern = r"(http\S+|www\.\S+)"
    email_pattern = r"\b[\w.-]+?@\w+?\.\w+?\b"

    replacements = {}
    counter = 0

    def replace_url(match):
        nonlocal counter
        key = f"[URL{counter}]"
        replacements[key] = match.group(0)
        counter += 1
        return key

    def replace_email(match):
        nonlocal counter
        key = f"[EMAIL{counter}]"
        replacements[key] = match.group(0)
        counter += 1
        return key

    text = re.sub(url_pattern, replace_url, text)
    text = re.sub(email_pattern, replace_email, text)

    return text, replacements

def unmask_entities(text: str, replacements: dict):
    for key, val in replacements.items():
        text = text.replace(key, val)
    return text

# Chunking using sentence tokenization
def chunk_text(text, max_tokens, tokenizer):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        candidate = (current_chunk + " " + sentence).strip()
        tokenized_len = len(tokenizer(candidate, return_tensors="pt")["input_ids"][0])

        if tokenized_len < max_tokens:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Language detection
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception as e:
        raise ValueError(f"Unable to detect language: {e}")

def load_marian_model(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    if model_name not in model_cache:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model_cache[model_name] = (tokenizer, model)
    return model_cache[model_name]

def load_m2m_model():
    model_name = "facebook/m2m100_418M"
    if model_name not in model_cache:
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        model_cache[model_name] = (tokenizer, model)
    return model_cache[model_name]

def validate_languages(src_lang, tgt_lang):
    try:
        _ = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}")
        return "marian"
    except:
        if src_lang in M2M_LANGUAGES and tgt_lang in M2M_LANGUAGES:
            return "m2m"
        else:
            raise ValueError(
                f"Translation not supported: source='{src_lang}' target='{tgt_lang}'. "
                "Try using ISO 639-1 codes. "
                f"Supported M2M100 languages: {sorted(list(M2M_LANGUAGES))}"
            )

def translate(text, src_lang, tgt_lang):
    text, replacements = preprocess_text(text)
    model_type = validate_languages(src_lang, tgt_lang)

    if model_type == "marian":
        tokenizer, model = load_marian_model(src_lang, tgt_lang)
        chunks = chunk_text(text, max_tokens=512, tokenizer=tokenizer)
        outputs = []
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
            translated = model.generate(**inputs)
            outputs.append(tokenizer.decode(translated[0], skip_special_tokens=True))
        result = " ".join(outputs)
        return unmask_entities(result, replacements)

    else:  # m2m
        tokenizer, model = load_m2m_model()
        tokenizer.src_lang = src_lang
        chunks = chunk_text(text, max_tokens=512, tokenizer=tokenizer)
        outputs = []
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
            generated = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
            outputs.append(tokenizer.decode(generated[0], skip_special_tokens=True))
        result = " ".join(outputs)
        return unmask_entities(result, replacements)
