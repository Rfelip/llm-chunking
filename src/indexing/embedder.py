import os
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
import faiss

class EmbeddingManager:
    def __init__(self, model_name="all-mpnet-base-v2", model_dir='data/models', index_dir='data/faiss_index'):
        self.model_name = model_name
        self.model_dir = model_dir
        self.index_dir = index_dir
        self.model = None
        self.index = None
        self.chunks = None
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)

    def load_or_download_model(self):
        """Load model from local directory or download if not exists"""
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True, 'convert_to_numpy' : True}
        model_path = os.path.join(self.model_dir, self.model_name)
        
        print("Loading model from local directory or downloading it...")
        self.model = HuggingFaceEmbeddings(cache_folder = model_path, model_name = self.model_name,
                                            encode_kwargs=encode_kwargs, model_kwargs = model_kwargs)
        return self.model

    def generate_embeddings(self, chunks):
        """Convert text chunks to embeddings"""
        if not self.model:
            self.load_or_download_model()
            
        print("Generating embeddings...")
        embbed = self.model.embed_documents(chunks)
        return embbed

    def create_faiss_index(self, embeddings):
        """Create and save FAISS index"""
        # Convert embeddings to float32 numpy array
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize embeddings
        embeddings = self.normalize_embeddings(embeddings)
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
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

    def generate_index_from_chunks(self, chunks=None, index_name = "index"):
        """Check for existing index, load if exists; else create from chunks and save."""
        index_path = os.path.join(self.index_dir, f"{index_name}.index")
        if os.path.exists(index_path):
            self.load_faiss_index(index_name)
            self.chunks = chunks
        else:
            if chunks is None:
                raise ValueError("Chunks must be provided to create a new index.")
            self.chunks = chunks
            embeddings = self.generate_embeddings(chunks)
            self.create_faiss_index(embeddings)
            self.save_faiss_index(index_name)
        return self.index
    
    def search_index(self, query, k = 3):
        query = np.array(query)
        query_vector = np.array(self.generate_embeddings(query))
        SIMILARITY, IDs = self.index.search(query_vector, k)
        resulting_chunks = self.chunks[IDs]
        return SIMILARITY, resulting_chunks