

import logging
import numpy as np
from PIL import Image
import cv2
from mtcnn import MTCNN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetector:
    
    
    def __init__(self):
        
        try:
            self.detector = MTCNN()
            logger.info("MTCNN face detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MTCNN detector: {e}")
            raise


def detect_and_preprocess(image_path, target_size=(160, 160)):
    
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image from {image_path}")
            return None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize detector
        detector = MTCNN()
        
        # Detect faces
        detections = detector.detect_faces(image_rgb)
        
        if not detections:
            logger.warning(f"No faces detected in {image_path}")
            return None
        
        # Get the first (largest) face
        face = detections[0]
        x, y, width, height = face['box']
        
        # Ensure coordinates are within image bounds
        x = max(0, x)
        y = max(0, y)
        width = min(width, image_rgb.shape[1] - x)
        height = min(height, image_rgb.shape[0] - y)
        
        # Extract face region
        face_pixels = image_rgb[y:y+height, x:x+width]
        
        # Resize to target size
        face_image = Image.fromarray(face_pixels)
        face_image = face_image.resize(target_size)
        face_array = np.asarray(face_image)
        
        # Normalize to [-1, 1] range (FaceNet requirement)
        face_pixels = face_array.astype('float32')
        face_pixels = (face_pixels - 127.5) / 127.5
        
        logger.info(f"Successfully processed face from {image_path}")
        return face_pixels
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None


if __name__ == "__main__":
    # Test the detector
    print("Testing Face Detector...")
    test_result = detect_and_preprocess("test_image.jpg")
    if test_result is not None:
        print(f"✅ Detection successful! Face shape: {test_result.shape}")
    else:
        print("❌ Detection failed or no face found")