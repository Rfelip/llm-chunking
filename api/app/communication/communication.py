from fastapi import APIRouter, HTTPException
from ..source.indexing_manager import IndexingManager
from ..source.conversation_manager import ConversationManager

router = APIRouter()

@router.post("/process_url")
async def process_url(url: str):
    manager = IndexingManager(url)
    await manager()
    return {"status": "index_created"}

@router.post("/ask")
async def ask_question(url: str, question: str, user: str = "default"):
    conv_manager = ConversationManager(url, user)
    conv_manager.index_manager = IndexingManager(url)
    response = await conv_manager.ask(question)
    return {"response": response}