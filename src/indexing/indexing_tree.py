from trafilatura import fetch_url, extract
from urllib.parse import urljoin, urlparse
from collections import deque


import re
import hashlib
import json
import aiofiles
import os

from asyncio import to_thread, get_event_loop, sleep
from pathlib import Path
from gzip import decompress, compress


def extract_info_from_website(url):
    download    = fetch_url(url)
    html_info   = extract(download, output_format="html",include_links=True, include_comments=False, include_tables=False, no_fallback=True)
    text_info   = extract(download, output_format="markdown", include_comments=False, include_tables=False, no_fallback=True)
    return html_info, text_info

class LinkNode:
    def __init__(self, url, depth=0, parent=None):
        self.url = url
        self.html_text = ""
        self.text = ""
        self.children = []
        self.depth = depth
        self.parent = parent

    async def populate_text(self):
        data_path = get_data_path(self.url)
        data_dir = data_path.parent
        if not await to_thread(os.path.exists, data_dir):
            await to_thread(os.makedirs, data_dir, exist_ok=True)

        if await data_exists(self.url):
            data = await load_data(self.url)
            self.text = data.get('text', '')
            self.html_text = data.get('html_text', '')
        else:
            loop = get_event_loop()
            html_text, page_text = await loop.run_in_executor(None, extract_info_from_website, self.url)
            self.text = page_text or ''
            self.html_text = html_text or ''
            await save_data(self.url, {'text': self.text, 'html_text': self.html_text})

class LinkTree:
    def __init__(self, root_url, max_depth=3):
        self.root = LinkNode(root_url)
        self.max_depth = max_depth
        self.visited = {root_url}
        self.queue = deque([self.root])

    def extract_links(self, html, base_url):
        pattern = r'href="(?!{}#|#)([^"]+)"'.format(re.escape(base_url))
        links = re.findall(pattern, html)
        return [urljoin(base_url, link) for link in links]

    async def build_tree(self):
        while self.queue:
            current_node = self.queue.popleft()
            
            if current_node.depth >= self.max_depth:
                continue
            try:
                await current_node.populate_text()
            except Exception as e:
                print(f"Error processing {current_node.url}: {e}")
                print("Information might be missing...")
                continue
            if not current_node.html_text:
                print(f"No HTML content for {current_node.url}.")
                continue
            
            child_links = self.extract_links(current_node.html_text, current_node.url)
            
            for link in child_links:
                if link not in self.visited:
                    self.visited.add(link)
                    child_node = LinkNode(
                       url=link,
                        depth=current_node.depth + 1,
                        parent=current_node
                    )
                    current_node.children.append(child_node)
                    self.queue.append(child_node)

    def print_tree(self):
        self._print_node(self.root)
    
    def _print_node(self, node, indent=0):
        print(' ' * indent * 4 + f'└── {node.url} (Depth: {node.depth})')
        for child in node.children:
            self._print_node(child, indent + 1)

def sanitize_url(url):
    """Sanitize the URL to make it safe for use in file paths."""
    # Parse the URL to get the netloc (domain) and path
    parsed_url = urlparse(url)
    netloc = parsed_url.netloc.replace(':', '_')  # Replace colons with underscores
    path = parsed_url.path.strip('/').replace('/', '_')  # Replace slashes with underscores
    
    # Combine netloc and path, and remove any remaining special characters
    sanitized = f"{netloc}_{path}" if path else netloc
    sanitized = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', sanitized)  # Replace any other special characters with underscores
    return sanitized

def get_data_path(url):
    """Generate a safe file path based on the sanitized URL."""
    sanitized_url = sanitize_url(url)
    hashed = hashlib.sha256(url.encode()).hexdigest()
    return Path('data') / "websites" / sanitized_url / f'{hashed}.json.gz'

async def data_exists(url):
    data_path = get_data_path(url)
    return await to_thread(os.path.exists, data_path)

async def load_data(url):
    '''Loads data using alternate threads to stop blocking behaviour.'''
    data_path = get_data_path(url)
    async with aiofiles.open(data_path, 'rb') as f:
        compressed_data = await f.read()
    data_bytes = await to_thread(decompress, compressed_data)
    return json.loads(data_bytes.decode('utf-8'))

async def save_data(url, data):
    data_path = get_data_path(url)
    lock_path = data_path.with_suffix('.lock')
    
    # Create lock file
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL)
            os.close(fd)
            break
        except FileExistsError:
            await sleep(0.1)
    
    try:
        # Compress in thread to avoid blocking
        data_str = json.dumps(data)
        data_bytes = data_str.encode('utf-8')
        compressed_data = await to_thread(compress, data_bytes)
        
        async with aiofiles.open(data_path, 'wb') as f:
            await f.write(compressed_data)
    finally:
        try:
            os.remove(lock_path)
        except:
            pass
async def build_index_from_url(url, depth = 1):
    index_tree = LinkTree(url, max_depth = depth)
    await index_tree.build_tree()
    sanitized_url = sanitize_url(url)
    folder_path = Path('data') / "websites" / sanitized_url 
    return index_tree, folder_path