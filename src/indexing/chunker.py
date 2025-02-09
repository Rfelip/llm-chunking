from .indexing_tree import build_index_from_url
from langchain.text_splitter import NLTKTextSplitter
import nltk
import os

nltk_data_dir = "../../data/ntlk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('punkt_tab', download_dir=nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

def build_chunks_from_tree(tree, chunk_size = 512):
    text_to_be_chunked = tree.root.text
    splitter = NLTKTextSplitter(chunk_size=chunk_size,
                                        chunk_overlap=20)
    
    docs = splitter.create_documents([text_to_be_chunked])
    return docs