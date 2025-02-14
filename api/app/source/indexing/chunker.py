import os
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_chunks_from_tree(tree, chunk_size = 384):
    text_to_be_chunked = tree.root.text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=28)
    docs = np.array(text_splitter.split_text(text_to_be_chunked))
    return docs