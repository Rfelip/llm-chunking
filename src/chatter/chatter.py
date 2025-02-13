import os
import json
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name):
    """Loads and prepares the model from local storage or downloads if missing"""
    model_path = f"data/models/{model_name}"
    
    if not os.path.exists(model_path):
        # Download and save model if not present locally
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
    else:
        # Load from local storage
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # For CPU-only systems
    model = model.to('cpu')
    return model, tokenizer

def save_conversation(website_name, query, answer):
    """Saves conversation history in a structured format"""
    conv_dir = f"data/conversations/websites/{website_name}"
    os.makedirs(conv_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_conversation.json"
    filepath = os.path.join(conv_dir, filename)
    
    # Save as JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'answer': answer
        }, f, ensure_ascii=False, indent=2)