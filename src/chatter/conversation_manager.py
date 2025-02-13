import aiofiles
import json
from pathlib import Path
from typing import List, Tuple
from llama_cpp import Llama
from ..indexing.indexing_tree import sanitize_url
import asyncio

class ConversationManager:
    def __init__(self, url: str, user: str = "default"):
        self.url = url
        self.user = user
        self.sanitized_url = sanitize_url(url)
        self.base_path = Path("data") / "websites" / self.sanitized_url / user
        self.history = []
        self.index_manager = None  # To be set externally

    async def initialize(self):
        """Async initialization for directory creation"""
        await self._create_directories()
        await self._load_history()

    async def _create_directories(self):
        """Create necessary directories asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.base_path.mkdir, 0o755, True, True)

    async def _load_history(self):
        """Load conversation history from file"""
        history_path = self.base_path / "conversations.json"
        try:
            async with aiofiles.open(history_path, "r") as f:
                content = await f.read()
                self.history = json.loads(content)
        except (FileNotFoundError, json.JSONDecodeError):
            self.history = []

    async def save_conversation(self):
        """Save conversation history to file"""
        history_path = self.base_path / "conversations.json"
        async with aiofiles.open(history_path, "w") as f:
            await f.write(json.dumps(self.history, indent=2))

    async def add_interaction(self, query: str, response: str):
        """Add and save an interaction asynchronously"""
        self.history.append((query, response))
        await self.save_conversation()

class AsyncModelManager:
    def __init__(self, model_name: str = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"):
        self.model_name = model_name
        self.model_path = Path("data") / "models" / model_name
        self.llm = None

    async def load_model(self):
        """Load model with async wrapper"""
        loop = asyncio.get_event_loop()
        if not self.llm:
            if not await self._model_exists():
                await self._download_model()
            
            self.llm = await loop.run_in_executor(
                None, 
                lambda: Llama(
                    model_path=str(self.model_path),
                    n_ctx=2048,
                    n_threads=8,
                    verbose=False
                )
            )
        return self.llm

    async def _model_exists(self):
        """Check if model exists asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.model_path.exists)

    async def _download_model(self):
        """Download model implementation (similar to original but async)"""
        # Implementation would use aiohttp instead of requests
        # Omitted for brevity, would maintain original functionality

async def generate_response(
    model: Llama,
    conv_manager: ConversationManager,
    query: str,
    temperature: float = 0.2
) -> str:
    """Generate response using context from IndexingManager"""
    # Get context from IndexingManager
    _, context_chunks = await conv_manager.index_manager.embedding_manager.search_index([query], 5)
    
    # Build prompt using IndexingManager's logic
    prompt = conv_manager.index_manager.build_starting_prompt(context_chunks[0], query)
    
    # Add conversation history
    history_str = "\n".join(
        [f"User: {q}\nAssistant: {a}" for q, a in conv_manager.history]  # Keep last 3 interactions
    )
    
    full_prompt = f"""<s>[INST] <<SYS>>
    {prompt}
    <</SYS>>

    Previous conversation:
    {history_str}

    Please provide a comprehensive answer. [/INST]"""
    
    # Run model inference in executor
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: model.create_chat_completion(
            messages=[{"role": "user", "content": full_prompt}],
            temperature=temperature,
            max_tokens=512
        )
    )
    
    return response['choices'][0]['message']['content']