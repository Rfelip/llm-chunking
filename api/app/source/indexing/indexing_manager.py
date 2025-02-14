import asyncio
from pathlib import Path
from .chunker import build_chunks_from_tree
from .embedder import EmbeddingManager
from .indexing_tree import LinkTree, sanitize_url

class IndexingManager:
    def __init__(self, url, model_name="all-mpnet-base-v2", chunk_size=384, max_depth=1):
        self.url = url
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.max_depth = max_depth
        self.sanitized_url = sanitize_url(url)
        self.index_dir = Path('data') / "websites" / self.sanitized_url
        self.embedding_manager = EmbeddingManager(
            model_name=model_name,
            index_dir=str(self.index_dir))
        self.tree = None
        self.chunks = None

    @staticmethod
    def build_starting_prompt(context_chunks, query):
        base_prompt = '''Using the information below, answer any question the user might have about this topic. If the answer cannot be found, write
"I'm sorry, but I couldn't find the answer."
'''
        for chunk in context_chunks:
            base_prompt += f"\nInformation: {chunk}"
        return base_prompt + f"\nUser question: {query}\nAnswer clearly and concisely."

    async def __call__(self, query=None):
        # Run full pipeline if not already executed
        if not self.tree:
            await self._execute_pipeline()

        # Handle query if provided
        if query is not None:
            return await self._handle_query(query)
        
        return self.embedding_manager.index

    async def _execute_pipeline(self):
        """Run all processing steps"""
        self.tree = LinkTree(self.url, max_depth=self.max_depth)
        await self.tree.build_tree()
        
        # Generate chunks
        loop = asyncio.get_event_loop()
        self.chunks = await loop.run_in_executor(
            None, build_chunks_from_tree, self.tree, self.chunk_size
        )
        
        # Create/save index
        await loop.run_in_executor(
            None, self.embedding_manager.generate_index_from_chunks, self.chunks, "main_index"
        )

    async def _handle_query(self, query):
        """Process user query and generate response"""
        loop = asyncio.get_event_loop()
        similarities, results = await loop.run_in_executor(
            None, self.embedding_manager.search_index, [query], 5
        )
        return self.build_starting_prompt(results[0], query)
    
    async def close(self):
        """Cleanup resources"""
        if hasattr(self, 'embedding_manager'):
            self.embedding_manager.close()