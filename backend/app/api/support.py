
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agents.support_agent import support_agent
from app.database.models import Conversation, Message
from app.database.session import get_db


router = APIRouter()


class SupportRequest(BaseModel):
    message: str = Field(min_length=3, max_length=4000)
    conversation_id: int | None = None


class SupportResponse(BaseModel):
    conversation_id: int
    reply: str


@router.post("/chat", response_model=SupportResponse)
async def chat_with_support(
    request: SupportRequest,
    db: Session = Depends(get_db)
):
    try:
        # Create a new conversation or retrieve an existing one
        if request.conversation_id is None:
            conversation = Conversation()
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
        else:
            conversation = db.get(
                Conversation,
                request.conversation_id
            )

            if conversation is None:
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found"
                )

        # Store the customer's message
        customer_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=request.message
        )

        db.add(customer_message)
        db.commit()


# Retrieve previous messages from this conversation
        previous_messages = (
            db.query(Message)
            .filter(
                Message.conversation_id == conversation.id,
                Message.id != customer_message.id
            )
            .order_by(Message.created_at, Message.id)
            .all()
        )

        # Build conversation context
        conversation_history = "\n".join(
            f"{'Customer' if msg.role == 'user' else 'Assistant'}: {msg.content}"
            for msg in previous_messages
        )

        # Include previous context only when available
        if conversation_history:
            agent_input = f"""
        Previous conversation:
        {conversation_history}

        Latest customer message:
        {request.message}

        Respond to the latest message using the previous conversation
        to understand the customer's context.
        """
        else:
            agent_input = request.message

        result = await support_agent.run(agent_input)

        # Store the AI response
        ai_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=result.output
        )

        db.add(ai_message)
        db.commit()

        return SupportResponse(
            conversation_id=conversation.id,
            reply=result.output
        )

    except HTTPException:
        raise

    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=502,
            detail=f"AI agent failed: {str(error)}"
        )


@router.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    conversation = db.get(Conversation, conversation_id)

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at, Message.id)
        .all()
    )

    return {
        "conversation_id": conversation.id,
        "messages": [
            {
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at
            }
            for message in messages
        ]
    }
