import json
from typing import List, AsyncGenerator, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from fastapi.responses import StreamingResponse

from ...api.deps import get_chat_interaction_use_case, get_collection_repository
from ...core.use_cases.chat_interaction import ChatInteractionUseCase
from ...core.domain.entities.chat import ChatSession, ChatMessage
from ...core.domain.exceptions import ChatSessionNotFoundError
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


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    use_case: ChatInteractionUseCase = Depends(get_chat_interaction_use_case),
):
    """
    Deletes chat session
    """
    try:
        await use_case.delete_session(session_id)
    except ChatSessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session with id '{session_id}' not found."
        )
    except Exception as e:
        # Catch any other unexpected errors during the deletion process
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while deleting the session: {str(e)}"
        )
    # FastAPI automatically returns 204 No Content for a successful response with no body
    # if the status_code is set.
    return


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


@router.api_route("/stream", methods=["GET", "POST"], status_code=200)
async def send_message_stream(
        body: MessageRequest = Body(None),
        session_id: str = Query(None),
        user_message: str = Query(None),
        use_case: ChatInteractionUseCase = Depends(get_chat_interaction_use_case),
):
    """
    Send a message and get streaming response.
    Returns a StreamingResponse where each chunk is JSON with fields 'response' and on first chunk 'retrieved_chunks'.
    """

    if body:
        session_id_val = body.session_id
        user_message_val = body.user_message
    else:
        session_id_val = session_id
        user_message_val = user_message

    if not session_id_val or not user_message_val:
        raise HTTPException(400, "Missing session_id or user_message")

    async def event_generator():
        try:
            async for chunk in use_case.process_streaming_message(
                    session_id=session_id_val,
                    user_message=user_message_val
            ):
                # Each chunk is a dict: {"response": "...", "retrieved_chunks": [...] (first only)}
                yield f"data: {json.dumps(chunk)}\n\n"
            # yield f"data: {json.dumps({'complete': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        # headers={
        #     "Cache-Control": "no-cache",
        #     "Connection": "keep-alive",
        #     "Access-Control-Allow-Origin": "*"
        # }
    )
