

__version__ = "1.0.0"
__author__ = "Face Verification Pipeline"

# Import main functions for easy access
from .detector import detect_and_preprocess
from .embeddings import get_embedding, batch_get_embeddings
from .verify import verify, batch_verify

__all__ = [
    'detect_and_preprocess',
    'get_embedding', 
    'batch_get_embeddings',
    'verify',
    'batch_verify'
]