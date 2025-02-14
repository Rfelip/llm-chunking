from fastapi import FastAPI, HTTPException, Body  # Add Body import
from source.indexing.indexing_manager import IndexingManager
from source.chatter.conversation_manager import ConversationManager, AsyncModelManager, generate_response
from source.indexing.indexing_tree import sanitize_url
import os
import logging

app = FastAPI()
active_indexes = {}

@app.post("/process_url")
async def process_url(url: str = Body(..., embed=True)): 
    manager = IndexingManager(url)
    await manager()  # Initialize the manager by running the pipeline
    active_indexes[manager.sanitized_url] = manager   # Store the manager instance
    return {"status": "index_created"}

@app.post("/ask")
async def ask_question( url: str = Body(..., embed=True),        # Parse from JSON
    query: str = Body(..., embed=True),      # All parameters must be
    user_id: str = Body(..., embed=True)     # wrapped with Body()
):
    sanitized_url = sanitize_url(url)  # Sanitize before checking):
    if sanitized_url not in active_indexes:
        raise HTTPException(418, "URL not indexed")
    
    manager = active_indexes[sanitized_url]  # Use sanitized key
    conv_manager = ConversationManager(url, user_id)
    conv_manager.index_manager = manager  # Link the IndexingManager
    await conv_manager.initialize()
    logging.info("Conversation context built, starting to load model...")
    # Initialize Model
    model_mgr = AsyncModelManager()
    model = await model_mgr.load_model()
    logging.info("Model loaded, starting to answer the question...")
    # Generate response
    response = await generate_response(model, conv_manager, query)
    
    # Save interaction
    await conv_manager.add_interaction(query, response)  # This saves automatically
    
    return {"response": response}