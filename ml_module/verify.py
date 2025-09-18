

import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify(emb1, emb2, threshold=0.6, method='cosine'):
    
    try:
        # Input validation
        if emb1 is None or emb2 is None:
            logger.error("One or both embeddings are None")
            return None
        
        if not isinstance(emb1, np.ndarray) or not isinstance(emb2, np.ndarray):
            logger.error("Embeddings must be numpy arrays")
            return None
        
        # Check if embeddings have same dimension
        if emb1.shape != emb2.shape:
            logger.error(f"Embedding shapes don't match: {emb1.shape} vs {emb2.shape}")
            return None
        
        # Support different embedding dimensions
        embedding_dim = emb1.shape[0]
        if embedding_dim not in [128, 512]:
            logger.warning(f"Unusual embedding dimension: {embedding_dim}")
        
        logger.debug(f"Comparing embeddings of dimension {embedding_dim}")
        
        # Calculate similarity based on method
        if method == 'cosine':
            # Cosine similarity (higher is more similar)
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            match = similarity >= threshold
            
        elif method == 'euclidean':
            # Euclidean distance (lower is more similar)
            distance = euclidean(emb1, emb2)
            
            # Convert distance to similarity score
            # For face_recognition (128-dim), typical distance range is 0-1.2
            # For FaceNet (512-dim normalized), distance range is 0-2
            if embedding_dim == 128:
                # face_recognition library typical range
                max_distance = 1.2
            else:
                # FaceNet typical range for normalized embeddings
                max_distance = 2.0
                
            similarity = max(0, 1 - (distance / max_distance))
            match = similarity >= threshold
            
        else:
            logger.error(f"Unknown method: {method}. Use 'cosine' or 'euclidean'")
            return None
        
        result = {
            "similarity": float(similarity),
            "match": bool(match),
            "method": method,
            "threshold": threshold,
            "embedding_dim": embedding_dim
        }
        
        logger.info(f"Verification complete: {method} similarity = {similarity:.4f}, match = {match} (dim={embedding_dim})")
        return result
        
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        return None


def verify_face_recognition_distance(emb1, emb2, threshold=0.6):
    
    try:
        import face_recognition
        
        # Calculate face distance (lower = more similar)
        distance = face_recognition.face_distance([emb1], emb2)[0]
        
        # Convert to similarity (higher = more similar)
        similarity = max(0, 1 - distance)
        match = distance <= threshold  # Note: for distance, lower is better
        
        result = {
            "similarity": float(similarity),
            "distance": float(distance),
            "match": bool(match),
            "method": "face_recognition_distance",
            "threshold": threshold
        }
        
        logger.info(f"face_recognition distance = {distance:.4f}, match = {match}")
        return result
        
    except ImportError:
        logger.error("face_recognition library not available")
        return None
    except Exception as e:
        logger.error(f"Error in face_recognition verification: {e}")
        return None


def batch_verify(emb_batch1, emb_batch2, threshold=0.6, method='cosine'):
    
    try:
        if emb_batch1 is None or emb_batch2 is None:
            logger.error("One or both embedding batches are None")
            return None
        
        if emb_batch1.shape != emb_batch2.shape:
            logger.error("Embedding batches must have same shape")
            return None
        
        results = []
        for i, (emb1, emb2) in enumerate(zip(emb_batch1, emb_batch2)):
            result = verify(emb1, emb2, threshold, method)
            if result is not None:
                result['pair_index'] = i
                results.append(result)
        
        logger.info(f"Batch verification complete: {len(results)} pairs processed")
        return results
        
    except Exception as e:
        logger.error(f"Error during batch verification: {e}")
        return None


if __name__ == "__main__":
    # Self-test
    print("Testing Updated Face Verification Module...")
    
    # Test with 128-dimensional embeddings (face_recognition)
    print("\n1. Testing with 128-dim embeddings:")
    emb1_128 = np.random.randn(128)
    emb2_128 = np.random.randn(128)
    
    result = verify(emb1_128, emb2_128, threshold=0.6, method='cosine')
    if result:
        print(f"   ✅ 128-dim test: {result['similarity']:.4f}, Match: {result['match']}")
    
    # Test with 512-dimensional embeddings (FaceNet)
    print("\n2. Testing with 512-dim embeddings:")
    emb1_512 = np.random.randn(512)
    emb2_512 = np.random.randn(512)
    
    result = verify(emb1_512, emb2_512, threshold=0.6, method='cosine')
    if result:
        print(f"   ✅ 512-dim test: {result['similarity']:.4f}, Match: {result['match']}")
    
    print("\n✅ All verification tests completed!")