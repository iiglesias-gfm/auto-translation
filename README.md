# 🌍 auto-translation API (2025 GFM Hackathon)

A simple yet flexible translation API:  
- Automatically detects the input language — no need to specify it.  
- Supports over 100 languages.  
- Choose between open-source transformer models or OpenAI’s LLM-powered translation.  
- Automatically selects the user’s browser language as the default output (when not provided).  
- Designed for scalability with full async support and non-blocking concurrency.

## 🛠 Setup

### 1. Clone and change directory

```bash
git clone https://github.com/iiglesias-gfm/auto-translation.git
cd auto-translation
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the API

```bash
uvicorn app.main:app --reload
```

Visit Swagger UI at: http://localhost:8000/docs

## 🧪 Try It

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{
        "text": "Please support my medical journey.",
        "src_lang": "en",
        "tgt_lang": "es",
        "campaign_id": "1234"
      }'
```

Response:

```json
{
  "translation": "Por favor apoya mi viaje médico."
}
```

## 🐳 Running with Docker (Optional)

### 1. Build the image

```bash
docker build -t translator-api .
```
### 2. Run the container
```bash
docker run -p 8000:8000 translator-api
```

This starts the API server at: http://localhost:8000
You can try it in your browser via the built-in Swagger UI at:

```bash
http://localhost:8000/docs
```
