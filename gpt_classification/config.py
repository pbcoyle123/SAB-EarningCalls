"""
Configuration file for API keys and settings
"""
import os
import json
from pathlib import Path

def load_config():
    """
    Load configuration from config.json file or environment variables.
    Priority: config.json > environment variables > default
    """
    config = {
        "openai_api_key": None
    }
    
    # Try to load from config.json file
    config_file = Path(__file__).parent / "config.json"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    
    # Override with environment variables if they exist
    if os.getenv('OPENAI_API_KEY'):
        config["openai_api_key"] = os.getenv('OPENAI_API_KEY')
    
    return config

def get_openai_api_key():
    """Get OpenAI API key from config"""
    config = load_config()
    api_key = config.get("openai_api_key")
    
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please either:\n"
            "1. Create a config.json file with your API key, or\n"
            "2. Set the OPENAI_API_KEY environment variable"
        )
    
    return api_key

def create_config_template():
    """Create a template config.json file"""
    template = {
        "openai_api_key": "your_openai_api_key_here"
    }
    
    config_file = Path(__file__).parent / "config.json"
    with open(config_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Created config template: {config_file}")
    print("Please edit config.json and add your OpenAI API key")

if __name__ == "__main__":
    # If run directly, create a template config file
    create_config_template()
