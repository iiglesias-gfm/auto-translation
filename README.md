🌍 auto-translation API (2025 GFM Hackathon)

A minimal FastAPI-based translation API using HuggingFace Transformers.

**Input:** A fundraiser campaign in any language  
**Output:** Translation to the target language

---

## 🛠 Setup

### 1. Clone and enter the directory

```bash
git clone https://github.com/your-username/text-translator-api.git
cd text-translator-api
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

---

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

---

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
