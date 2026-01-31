#!/usr/bin/env python3
"""
Chat Microservice for KRX Sector Rotation Dashboard
Standalone service deployable to Railway with automatic HTTPS

Provides AI-powered investment Q&A using:
- Gemini-2.0-Flash via OpenRouter
- PostgreSQL for conversation history
- Security-hardened system prompt
"""

import os
import uuid
import httpx
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, selectinload
from sqlalchemy import String, Text, ForeignKey, select, desc, DateTime
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
import uuid as uuid_module


# ============================================
# Database Models
# ============================================

class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid_module.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid_module.uuid4
    )
    user_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        lazy="selectin"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid_module.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid_module.uuid4
    )
    conversation_id: Mapped[uuid_module.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String(20))  # 'user' or 'assistant'
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow
    )

    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="messages"
    )


# ============================================
# Database Setup
# ============================================

DATABASE_URL = os.getenv("DATABASE_URL", "")

# Railway uses postgres://, SQLAlchemy needs postgresql+asyncpg://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = None
async_session_maker = None


async def init_db():
    """Initialize database and create tables"""
    global engine, async_session_maker
    if not DATABASE_URL:
        print("WARNING: DATABASE_URL not set")
        return

    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session_maker = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database initialized successfully")


async def close_db():
    """Close database connections"""
    global engine
    if engine:
        await engine.dispose()


async def get_session() -> AsyncSession:
    """Get database session"""
    if async_session_maker is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    async with async_session_maker() as session:
        yield session


# ============================================
# Configuration
# ============================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "google/gemini-2.0-flash-001"

# Security-hardened system prompt
SYSTEM_PROMPT = """You are a KRX Sector Rotation investment assistant for Wealth PBs (Private Bankers).

## STRICT SECURITY RULES (MUST FOLLOW):
1. ONLY answer questions about stock recommendations, themes, market conditions, and dashboard navigation
2. NEVER explain internal algorithms or methodologies:
   - Do NOT explain Fiedler eigenvalue calculation
   - Do NOT explain HMM (Hidden Markov Model) regime detection
   - Do NOT explain correlation matrix computation
   - Do NOT explain PageRank centrality calculation
   - Do NOT explain meta-labeling model internals
   - Do NOT explain Bollinger Band technical details
3. If asked about implementation/algorithms, respond: "이 정보는 내부 분석 시스템에서 제공됩니다. 결과만 안내해 드릴 수 있습니다."
4. Stay within the scope of pre-computed analysis data
5. Redirect users to appropriate dashboard pages when relevant
6. NEVER make up stock prices, dates, or specific numbers not in the context
7. If you don't have specific data, say so honestly

## VOCABULARY (use consistently):
- 군집성 (Cohesion): 테마 내 종목들의 동조화 강도 (높을수록 함께 움직임)
- 모멘텀 (Momentum): 상승 추세 + 매수 조건을 충족한 종목 (Transition regime + Above BB)
- 시그널 (Signal): 매수/매도 신호 (메타 레이블링 품질 필터 통과 여부)
- 핵심 종목 (Key Player): 테마 내 중심성이 높은 종목
- TIER 1: 최고 품질 테마 (메타 레이블링 필터 통과)

## DASHBOARD PAGES (guide users here):
- 개요 (Overview): Main dashboard - 전체 현황, 모멘텀 종목, 테마 건강도
- 모멘텀 (Momentum): /breakout.html - 관심 종목 리스트, 단계별 분포
- 시그널 (Signals): /signals.html - 테마별 신호 품질, 통과율
- 군집성 (Cohesion): /cohesion.html - 테마 군집성 분석
- 네트워크 (Network): /theme-graph.html - 테마 관계도 시각화

## RESPONSE STYLE:
- Answer in the user's language (Korean or English)
- Keep responses concise and actionable
- Use bullet points and tables when appropriate
- Always cite which dashboard page has more details
- Be professional and helpful

## ABOUT THIS SYSTEM:
The KRX Sector Rotation Dashboard analyzes Korean stock market themes using:
- Theme cohesion analysis (how stocks in a theme move together)
- Regime detection (Bull/Bear market states)
- Trend analysis (momentum stages)
- Signal quality filtering

You help users understand the dashboard outputs and make investment decisions based on the analysis results.
"""


# ============================================
# App Lifecycle
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    await init_db()
    yield
    await close_db()


# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="KRX Chat Service",
    description="AI-powered investment Q&A for KRX Sector Rotation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS - Allow all origins for widget embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Request/Response Models
# ============================================

class ChatMessageRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    language: str = "ko"


class ChatMessageResponse(BaseModel):
    response: str
    conversation_id: str
    message_id: str


class ConversationListItem(BaseModel):
    id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int


class MessageItem(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime


class ConversationHistory(BaseModel):
    id: str
    title: Optional[str]
    messages: List[MessageItem]


# ============================================
# Helper Functions
# ============================================

async def call_openrouter(messages: list, language: str = "ko") -> str:
    """Call OpenRouter API with Gemini model"""
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY not configured"
        )

    api_messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ] + messages

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://krx-chat.up.railway.app"),
        "X-Title": os.getenv("OPENROUTER_SITE_NAME", "KRX-Sector-Rotation-Chat"),
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": api_messages,
        "max_tokens": 1000,
        "temperature": 0.3,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"OpenRouter API error: {e.response.text}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to call OpenRouter: {str(e)}"
            )


def generate_title(message: str) -> str:
    """Generate a short title from the first message"""
    title = message.strip()[:50]
    if len(message) > 50:
        title += "..."
    return title


# ============================================
# API Endpoints
# ============================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "KRX Chat Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/api/chat/message",
            "health": "/api/chat/health"
        }
    }


@app.post("/api/chat/message", response_model=ChatMessageResponse)
async def send_message(
    request: ChatMessageRequest,
    session: AsyncSession = Depends(get_session)
):
    """Send a message and get AI response"""
    try:
        conversation = None

        if request.conversation_id:
            try:
                conv_uuid = uuid.UUID(request.conversation_id)
                result = await session.execute(
                    select(Conversation)
                    .options(selectinload(Conversation.messages))
                    .where(Conversation.id == conv_uuid)
                )
                conversation = result.scalar_one_or_none()
            except ValueError:
                pass

        if not conversation:
            conversation = Conversation(
                title=generate_title(request.message)
            )
            session.add(conversation)
            await session.flush()

        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=request.message
        )
        session.add(user_message)

        result = await session.execute(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at.desc())
            .limit(10)
        )
        history_messages = list(reversed(result.scalars().all()))

        api_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in history_messages
        ]
        api_messages.append({"role": "user", "content": request.message})

        ai_response = await call_openrouter(api_messages, request.language)

        assistant_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=ai_response
        )
        session.add(assistant_message)

        conversation.updated_at = datetime.utcnow()

        await session.commit()

        return ChatMessageResponse(
            response=ai_response,
            conversation_id=str(conversation.id),
            message_id=str(assistant_message.id)
        )

    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/new")
async def new_conversation(
    session: AsyncSession = Depends(get_session)
):
    """Start a new conversation"""
    try:
        conversation = Conversation()
        session.add(conversation)
        await session.commit()

        return {
            "conversation_id": str(conversation.id),
            "created_at": conversation.created_at.isoformat()
        }
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/history/{conversation_id}", response_model=ConversationHistory)
async def get_conversation_history(
    conversation_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Get full conversation history"""
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation ID")

    result = await session.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conv_uuid)
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    sorted_messages = sorted(conversation.messages, key=lambda m: m.created_at)

    return ConversationHistory(
        id=str(conversation.id),
        title=conversation.title,
        messages=[
            MessageItem(
                id=str(msg.id),
                role=msg.role,
                content=msg.content,
                created_at=msg.created_at
            )
            for msg in sorted_messages
        ]
    )


@app.get("/api/chat/conversations", response_model=List[ConversationListItem])
async def list_conversations(
    limit: int = Query(20, ge=1, le=100),
    session: AsyncSession = Depends(get_session)
):
    """List recent conversations"""
    result = await session.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .order_by(desc(Conversation.updated_at))
        .limit(limit)
    )
    conversations = result.scalars().all()

    return [
        ConversationListItem(
            id=str(conv.id),
            title=conv.title,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            message_count=len(conv.messages)
        )
        for conv in conversations
    ]


@app.delete("/api/chat/conversation/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Delete a conversation and all its messages"""
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation ID")

    result = await session.execute(
        select(Conversation).where(Conversation.id == conv_uuid)
    )
    conversation = result.scalar_one_or_none()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await session.delete(conversation)
    await session.commit()

    return {"status": "deleted", "conversation_id": conversation_id}


@app.get("/api/chat/health")
async def chat_health():
    """Health check for chat service"""
    return {
        "status": "healthy",
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "database_configured": bool(DATABASE_URL),
        "model": MODEL
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
