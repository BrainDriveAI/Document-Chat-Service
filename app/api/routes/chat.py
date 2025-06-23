# File: app/api/routes/chat.py

from typing import List, AsyncGenerator, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from ...api.deps import get_chat_interaction_use_case, get_collection_repository
from ...core.use_cases.chat_interaction import ChatInteractionUseCase
from ...core.domain.entities.chat import ChatSession, ChatMessage
from pydantic import BaseModel, Field

router = APIRouter()


class CreateSessionRequest(BaseModel):
    name: str = Field(..., min_length=1)
    collection_id: Optional[str] = None


class ChatSessionResponse(BaseModel):
    id: str
    name: str
    collection_id: Optional[str]
    created_at: str
    updated_at: str
    message_count: int


def to_response_session_model(domain: ChatSession) -> ChatSessionResponse:
    return ChatSessionResponse(
        id=domain.id,
        name=domain.name,
        collection_id=domain.collection_id,
        created_at=domain.created_at.isoformat(),
        updated_at=domain.updated_at.isoformat(),
        message_count=domain.message_count
    )


@router.get("/sessions", response_model=List[ChatSessionResponse])
async def list_chat_sessions(
    use_case: ChatInteractionUseCase = Depends(get_chat_interaction_use_case),
):
    """
    List all chat sessions.
    """
    try:
        chat_sessions = await use_case.list_sessions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list chat sessions: {str(e)}")
    return [to_response_session_model(c) for c in chat_sessions]


@router.post("/sessions", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
        body: CreateSessionRequest,
        use_case: ChatInteractionUseCase = Depends(get_chat_interaction_use_case),
        collection_repo=Depends(get_collection_repository),
):
    # If collection_id provided, verify exists
    if body.collection_id:
        coll = await collection_repo.find_by_id(body.collection_id)
        if not coll:
            raise HTTPException(status_code=404, detail=f"Collection {body.collection_id} not found")
    session = await use_case.create_session(name=body.name, collection_id=body.collection_id)
    return ChatSessionResponse(
        id=session.id,
        name=session.name,
        collection_id=session.collection_id,
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
        message_count=session.message_count
    )


class MessageRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    user_message: str = Field(..., min_length=1)


class MessageResponse(BaseModel):
    id: str
    session_id: str
    user_message: str
    assistant_response: str
    retrieved_chunks: List[str]
    created_at: str
    response_time_ms: int


def to_response_message_model(domain: ChatMessage) -> MessageResponse:
    return MessageResponse(
        id=domain.id,
        session_id=domain.session_id,
        user_message=domain.user_message,
        assistant_response=domain.assistant_response,
        retrieved_chunks=domain.retrieved_chunks,
        created_at=domain.created_at.isoformat(),
        response_time_ms=domain.response_time_ms
    )


@router.get("/messages", response_model=List[MessageResponse])
async def list_chat_messages(
    session_id: str = None,
    use_case: ChatInteractionUseCase = Depends(get_chat_interaction_use_case),
):
    """
    List all chat messages for a session.
    """
    try:
        chat_messages = await use_case.list_chat_messages(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list chat session messages: {str(e)}")
    return [to_response_message_model(c) for c in chat_messages]


@router.post("/message", response_model=MessageResponse)
async def send_message(
        body: MessageRequest,
        use_case: ChatInteractionUseCase = Depends(get_chat_interaction_use_case),
):
    """
    Send a message and get full response (non-streaming).
    """
    try:
        chat_msg = await use_case.process_message(
            session_id=body.session_id,
            user_message=body.user_message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")
    return MessageResponse(
        id=chat_msg.id,
        session_id=chat_msg.session_id,
        user_message=chat_msg.user_message,
        assistant_response=chat_msg.assistant_response,
        retrieved_chunks=chat_msg.retrieved_chunks,
        created_at=chat_msg.created_at.isoformat(),
        response_time_ms=chat_msg.response_time_ms
    )


@router.post("/stream", status_code=200)
async def send_message_stream(
        body: MessageRequest,
        use_case: ChatInteractionUseCase = Depends(get_chat_interaction_use_case),
):
    """
    Send a message and get streaming response.
    Returns a StreamingResponse where each chunk is JSON with fields 'response' and on first chunk 'retrieved_chunks'.
    """

    async def event_generator():
        try:
            async for chunk in use_case.process_streaming_message(
                    session_id=body.session_id,
                    user_message=body.user_message
            ):
                # Each chunk is a dict: {"response": "...", "retrieved_chunks": [...] (first only)}
                yield (f"data: {chunk}\n\n")
            # After streaming done, no further action here; saving was handled in non-streaming process?
            # Actually, in streaming flow, ChatInteractionUseCase does not save the final message.
            # You may want to gather the full response then save. For simplicity, client can call non-streaming if persistence is needed.
        except Exception as e:
            # On error, send as data
            yield (f"data: {{'error': '{str(e)}'}}\n\n")

    return StreamingResponse(event_generator(), media_type="text/event-stream")
