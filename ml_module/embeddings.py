"""
Fixed Face Embedding Generation Module

Using face_recognition library instead of keras-facenet for reliable embeddings.
"""

import logging
import numpy as np
import face_recognition
from PIL import Image
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global flag to check if face_recognition is available
FACE_RECOGNITION_AVAILABLE = True

try:
    # Test face_recognition import
    import face_recognition
    logger.info("face_recognition library initialized successfully")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logger.error("face_recognition library not available. Install with: pip install face_recognition")


def get_embedding_face_recognition(face_pixels):
    
    if not FACE_RECOGNITION_AVAILABLE:
        logger.error("face_recognition library not available")
        return None
        
    try:
        # Convert to uint8 if needed
        if face_pixels.dtype != np.uint8:
            if face_pixels.min() >= -1 and face_pixels.max() <= 1:
                # Denormalize from [-1,1] to [0,255]
                face_uint8 = ((face_pixels + 1) * 127.5).astype(np.uint8)
            else:
                face_uint8 = (face_pixels * 255).astype(np.uint8) if face_pixels.max() <= 1 else face_pixels.astype(np.uint8)
        else:
            face_uint8 = face_pixels
            
        logger.debug(f"Processing face with shape: {face_uint8.shape}")
        
        # Get face encodings
        encodings = face_recognition.face_encodings(face_uint8)
        
        if len(encodings) == 0:
            logger.warning("No face encodings found in the image")
            return None
            
        # Return first encoding
        encoding = encodings[0]
        logger.info(f"Generated face_recognition embedding with shape: {encoding.shape}")
        
        return encoding
        
    except Exception as e:
        logger.error(f"Error generating face_recognition embedding: {e}")
        return None


def get_embedding_alternative_facenet():
    """Alternative FaceNet implementation using TensorFlow Hub."""
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
        
        # Load FaceNet model from TensorFlow Hub
        model_url = "https://tfhub.dev/google/facenet/1"
        facenet = hub.load(model_url)
        
        def generate_embedding(face_pixels):
            # Ensure correct input format
            if face_pixels.shape != (160, 160, 3):
                face_pixels = cv2.resize(face_pixels, (160, 160))
            
            # Add batch dimension
            face_batch = tf.expand_dims(face_pixels, 0)
            face_batch = tf.cast(face_batch, tf.float32)
            
            # Generate embedding
            embedding = facenet(face_batch)
            return embedding.numpy()[0]
            
        return generate_embedding
        
    except ImportError:
        logger.error("TensorFlow Hub not available")
        return None
    except Exception as e:
        logger.error(f"Error loading alternative FaceNet: {e}")
        return None


def get_embedding(face_pixels):
    
    try:
        # Input validation
        if face_pixels is None:
            logger.error("Input face_pixels is None")
            return None
        
        if not isinstance(face_pixels, np.ndarray):
            logger.error("Input must be numpy array")
            return None
        
        # Try face_recognition library first (most reliable)
        if FACE_RECOGNITION_AVAILABLE:
            embedding = get_embedding_face_recognition(face_pixels)
            if embedding is not None:
                return embedding
        
        # Fallback: warn user about installation
        logger.error("Could not generate embeddings. Please install face_recognition library:")
        logger.error("pip install face_recognition")
        logger.error("Note: On Windows, you might need: pip install cmake dlib face_recognition")
        
        return None
        
    except Exception as e:
        logger.error(f"Error in get_embedding: {e}")
        return None


def batch_get_embeddings(face_batch):
    
    try:
        if face_batch is None or len(face_batch) == 0:
            logger.error("Empty face batch")
            return None
        
        embeddings = []
        for i, face in enumerate(face_batch):
            embedding = get_embedding(face)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                logger.warning(f"Failed to generate embedding for face {i}")
        
        if len(embeddings) == 0:
            logger.error("No embeddings generated")
            return None
        
        return np.array(embeddings)
        
    except Exception as e:
        logger.error(f"Error in batch embedding generation: {e}")
        return None


if __name__ == "__main__":
    # Self-test
    print("Testing Fixed Face Embedding Module...")
    
    if not FACE_RECOGNITION_AVAILABLE:
        print("❌ face_recognition library not available")
        print("Install with: pip install face_recognition")
        print("On Windows: pip install cmake dlib face_recognition")
        exit(1)
    
    # Create test face data
    dummy_face1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    dummy_face2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Test embedding generation
    print("Testing embedding generation...")
    emb1 = get_embedding(dummy_face1)
    emb2 = get_embedding(dummy_face2)
    
    if emb1 is not None and emb2 is not None:
        print(f"✅ Embeddings generated!")
        print(f"   Embedding 1 shape: {emb1.shape}")
        print(f"   Embedding 2 shape: {emb2.shape}")
        
        # Check if embeddings are different
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"   Similarity: {similarity:.6f}")
        
        if np.allclose(emb1, emb2):
            print("   ⚠️  Embeddings are identical (this shouldn't happen)")
        else:
            print("   ✅ Embeddings are different (good!)")
    else:
        print("❌ Embedding generation failed")