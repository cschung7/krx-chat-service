# KRX Chat Service

AI-powered investment Q&A microservice for KRX Sector Rotation Dashboard.

## Features

- 🤖 Gemini-2.0-Flash via OpenRouter
- 💬 Conversation history with PostgreSQL
- 🔒 Security-hardened system prompt
- 🌐 CORS enabled for widget embedding

## Deploy to Railway

1. Click **New Project** in Railway dashboard
2. Select **Deploy from GitHub repo**
3. Add these environment variables:
   - `DATABASE_URL` - Add PostgreSQL plugin (Railway provides this automatically)
   - `OPENROUTER_API_KEY` - Your OpenRouter API key
4. Deploy!

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string (auto-provided by Railway) |
| `OPENROUTER_API_KEY` | OpenRouter API key for Gemini |
| `PORT` | Server port (auto-provided by Railway) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat/message` | POST | Send message, get AI response |
| `/api/chat/history/{id}` | GET | Get conversation history |
| `/api/chat/conversations` | GET | List recent conversations |
| `/api/chat/health` | GET | Health check |

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost/chat"
export OPENROUTER_API_KEY="sk-or-..."

# Run server
uvicorn main:app --reload
```
