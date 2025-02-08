from trafilatura import fetch_url, extract
from urllib.parse import urljoin
from collections import deque
import re

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

    def populate_text(self):
        html_text, page_text = extract_info_from_website(self.url)
        self.text = page_text
        self.html_text = html_text

class LinkTree:

    def __init__(self, root_url, max_depth=3):
        
        self.root = LinkNode(root_url)

        self.max_depth = max_depth

        self.visited = {root_url}

        self.queue = deque([self.root])

    def extract_links(self, html, base_url):
        pattern = r'href="(?!{}#|#)([^"]+)"'.format(re.escape(base_url))
        links = re.findall(pattern, html)
        # Convert relative links to absolute URLs
        return [urljoin(base_url, link) for link in links]

    
    
    def build_tree(self):
        while self.queue:
            current_node = self.queue.popleft()
            
            if current_node.depth >= self.max_depth:
                continue
            
            try:
                current_node.populate_text()
            except:
                print("When processing one of the nodes, the extraction of text didn't work.")
                print("Information might be missing...")
                continue
            
            if not current_node.html_text:
                print("One of the nodes didn't have any info.")
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

def build_index_from_url(url):
    index_tree = LinkTree(url, max_depth = 1)
    index_tree.build_tree()
    return index_tree