"""
Environment variables for Azure OpenAI and other external services.
These are used by the utils/openai_utils.py and utils/text_utils.py modules.
"""
import os

# Text processing limits
FULL_TEXT_TOKEN_LIMIT = int(os.getenv("FULL_TEXT_TOKEN_LIMIT", "8000"))

# Tenacity retry settings
TENACITY_STOP_AFTER_DELAY = int(os.getenv("TENACITY_STOP_AFTER_DELAY", "300"))
TENACITY_TIMEOUT = int(os.getenv("TENACITY_TIMEOUT", "120"))

# Azure OpenAI Vision API settings
AZURE_OPENAI_VISION_API_VERSION = os.getenv("AZURE_OPENAI_VISION_API_VERSION", "2024-02-15-preview")
AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT", "")
AZURE_VISION_KEY = os.getenv("AZURE_VISION_KEY", "")
