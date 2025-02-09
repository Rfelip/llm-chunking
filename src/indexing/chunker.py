from .indexing_tree import build_index_from_url
from langchain.text_splitter import MarkdownTextSplitter

def build_chunks_from_tree(tree, chunk_size = 256):
    text_to_be_chunked = tree.root.text
    markdown_splitter = MarkdownTextSplitter(chunk_size=chunk_size,
                                             chunk_overlap=256,
                                             strip_whitespace=True)
    docs = markdown_splitter.create_documents([text_to_be_chunked])
    return docs