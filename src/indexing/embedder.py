import os
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
import faiss

class EmbeddingManager:
    def __init__(self, model_name='all-MiniLM-L6-v2', model_dir='data/models', index_dir='data/faiss_index'):
        self.model_name = model_name
        self.model_dir = model_dir
        self.index_dir = index_dir
        self.model = None
        self.index = None

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)

    def load_or_download_model(self):
        """Load model from local directory or download if not exists"""
        model_path = os.path.join(self.model_dir, self.model_name)
        
        if os.path.exists(model_path):
            print("Loading model from local directory...")
            self.model = HuggingFaceEmbeddings(cache_folder = model_path, model_name = self.model_name)
        else:
            print("Downloading model...")
            self.model = HuggingFaceEmbeddings(cache_folder = model_path, model_name = self.model_name)
            self.model.save(model_path)
        return self.model

    async def generate_embeddings(self, chunks):
        """Convert text chunks to embeddings"""
        if not self.model:
            self.load_or_download_model()
            
        print("Generating embeddings...")
        return await self.model.aembed_documents(chunks)

    def normalize_embeddings(self, embeddings):
        """Normalize embeddings to unit length"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    def create_faiss_index(self, embeddings):
        """Create and save FAISS index"""
        # Convert embeddings to float32 numpy array
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize embeddings
        embeddings = self.normalize_embeddings(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        M         = 32
        self.index = faiss.IndexHNSWFlat(dimension)
        self.index.add(embeddings)
        print(f"Created FAISS index with {self.index.ntotal} vectors")
        return self.index

    def save_faiss_index(self, index_name='my_index'):
        """Save FAISS index to disk"""
        if self.index is None:
            raise ValueError("Index not initialized. Create index first.")
            
        index_path = os.path.join(self.index_dir, f"{index_name}.index")
        faiss.write_index(self.index, index_path)
        print(f"Index saved to {index_path}")

    def load_faiss_index(self, index_name='my_index'):
        """Load FAISS index from disk"""
        index_path = os.path.join(self.index_dir, f"{index_name}.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No index found at {index_path}")
            
        self.index = faiss.read_index(index_path)
        print(f"Loaded index with {self.index.ntotal} vectors")
        return self.index

    async def __call__(self, chunks=None, index_name):
        """Check for existing index, load if exists; else create from chunks and save."""
        index_path = os.path.join(self.index_dir, f"{index_name}.index")
        if os.path.exists(index_path):
            self.load_faiss_index(index_name)
        else:
            if chunks is None:
                raise ValueError("Chunks must be provided to create a new index.")
            
            text_chunks = [doc.page_content for doc in chunks]
            embeddings = await self.generate_embeddings(text_chunks)
            self.create_faiss_index(embeddings)
            self.save_faiss_index(index_name)
        return self.index