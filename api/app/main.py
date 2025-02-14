from fastapi import FastAPI, HTTPException
from source.indexing.indexing_manager import IndexingManager
from source.chatter.conversation_manager import ConversationManager, AsyncModelManager, generate_response
import os
import logging

app = FastAPI()
active_indexes = {}

@app.post("/process_url")
async def process_url(url: str):
    manager = IndexingManager(url)
    index = await manager()
    active_indexes[url] = index
    return {"status": "index_created"}

@app.post("/ask")
async def ask_question(url: str, query: str, user_id: str):
    if url not in active_indexes:
        raise HTTPException(status_code=404, detail="URL not indexed")

    manager = active_indexes[url]
    conv_manager = ConversationManager(url, user_id)
    
    # Perform search
    similarities, results = manager.embedding_manager.search_index([query], 3)
    
    
    await conv_manager.initialize()

    # Initialize Model
    model_mgr = AsyncModelManager()
    model = await model_mgr.load_model()

    # Example interaction
    response = await generate_response(model, conv_manager, query)
    
    await conv_manager.add_interaction(query, response)
    # Save conversation
    await conv_manager.save_conversation(query, response)
    
    return {"response": response}