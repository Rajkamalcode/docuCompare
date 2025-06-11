import os

# MongoDB Configuration
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.environ.get("MONGO_DB", "document_processor")
MONGO_COLLECTION = os.environ.get("MONGO_COLLECTION", "documents")

# Vertex AI Configuration
VERTEX_AI_PROJECT_ID = os.environ.get("VERTEX_AI_PROJECT_ID", "cifcl-poc-ai")
VERTEX_AI_LOCATION = os.environ.get("VERTEX_AI_LOCATION", "asia-south1")
VERTEX_AI_MODEL = os.environ.get("VERTEX_AI_MODEL", "gemini-1.5-pro-002")
CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "database/cifcl-poc-ai.json")

# Document Storage
DOCUMENTS_FOLDER = os.environ.get("DOCUMENTS_FOLDER", "documents")

# Comparison Configuration
EXACT_MATCH_THRESHOLD = 1.0  # For exact string matching
SEMANTIC_MATCH_THRESHOLD = 0.85  # For semantic matching

# Ensure documents folder exists
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
