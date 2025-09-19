"""
ML Module Interface for Server Integration
Handles enrollment and verification of faces
"""

import os
import json
import logging
from datetime import datetime
from typing import Tuple, Optional, List, Union
from io import BytesIO
import numpy as np

from .detector import detect_and_preprocess
from .embeddings import get_embedding
from .verify import verify
from database.models import Student
from attendance.db import get_session

logger = logging.getLogger(__name__)

class MLInterface:
    """Interface class for ML operations"""
    
    def __init__(self, upload_folder: str = "uploads"):
        self.upload_folder = upload_folder
        self.enrollment_folder = os.path.join(upload_folder, "enrollment")
        self.attendance_folder = os.path.join(upload_folder, "attendance")
        
        # Create directories
        os.makedirs(self.enrollment_folder, exist_ok=True)
        os.makedirs(self.attendance_folder, exist_ok=True)
        
    def enroll_student(self, student_id: str, image_files: List) -> Tuple[bool, Optional[str]]:
        """
        Enroll a student with their face images
        
        Args:
            student_id: Unique student identifier
            image_files: List of uploaded image files
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            if not image_files:
                return False, "No images provided"
            
            session = get_session()
            student = session.query(Student).filter_by(student_id=student_id).first()
            
            if not student:
                return False, f"Student {student_id} not found in database"
            
            # Process the first valid image for enrollment
            for image_file in image_files:
                try:
                    # Save uploaded image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{student_id}_enrollment_{timestamp}.jpg"
                    image_path = os.path.join(self.enrollment_folder, filename)
                    
                    # Save file to disk
                    image_file.save(image_path)
                    
                    # Process face
                    face_pixels = detect_and_preprocess(image_path)
                    if face_pixels is None:
                        logger.warning(f"No face detected in enrollment image for {student_id}")
                        continue
                    
                    # Generate embedding
                    embedding = get_embedding(face_pixels)
                    if embedding is None:
                        logger.warning(f"Failed to generate embedding for {student_id}")
                        continue
                    
                    # Store embedding in database
                    embedding_json = json.dumps(embedding.tolist())
                    student.face_embedding = embedding_json
                    student.enrolled_photo_path = image_path
                    student.is_enrolled = True
                    student.updated_at = datetime.now()
                    
                    session.commit()
                    session.close()
                    
                    logger.info(f"Successfully enrolled student {student_id}")
                    return True, None
                    
                except Exception as e:
                    logger.error(f"Error processing enrollment image for {student_id}: {e}")
                    continue
            
            session.close()
            return False, "No valid face found in any provided image"
            
        except Exception as e:
            logger.error(f"Error in student enrollment: {e}")
            return False, str(e)
    
    def verify_face(self, student_id: str, selfie_file: Union[BytesIO, object]) -> Tuple[bool, float, bool, Optional[str]]:
        """
        Verify a student's face against their enrolled embedding
        
        Args:
            student_id: Student identifier
            selfie_file: Uploaded selfie file (BytesIO or file object)
            
        Returns:
            Tuple of (face_match, confidence_score, liveness_passed, error_message)
        """
        try:
            session = get_session()
            student = session.query(Student).filter_by(student_id=student_id).first()
            
            if not student or not student.is_enrolled:
                session.close()
                return False, 0.0, False, f"Student {student_id} not enrolled"
            
            # Load enrolled embedding
            if not student.face_embedding:
                session.close()
                return False, 0.0, False, "No face embedding found for student"
            
            enrolled_embedding = np.array(json.loads(student.face_embedding))
            
            # Save selfie temporarily
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            temp_filename = f"{student_id}_verify_{timestamp}.jpg"
            temp_path = os.path.join(self.attendance_folder, temp_filename)
            
            # Handle different file types
            if hasattr(selfie_file, 'save'):
                # Flask FileStorage object
                selfie_file.save(temp_path)
            elif isinstance(selfie_file, BytesIO):
                # BytesIO from base64
                with open(temp_path, 'wb') as f:
                    f.write(selfie_file.getvalue())
            else:
                session.close()
                return False, 0.0, False, "Invalid selfie file format"
            
            # Process selfie
            face_pixels = detect_and_preprocess(temp_path)
            if face_pixels is None:
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                session.close()
                return False, 0.0, False, "No face detected in selfie"
            
            # Generate embedding for selfie
            selfie_embedding = get_embedding(face_pixels)
            if selfie_embedding is None:
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                session.close()
                return False, 0.0, False, "Failed to generate embedding from selfie"
            
            # Perform verification
            result = verify(enrolled_embedding, selfie_embedding, threshold=0.6, method='cosine')
            if result is None:
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                session.close()
                return False, 0.0, False, "Face verification failed"
            
            # Simple liveness check (placeholder - you can enhance this)
            liveness_passed = self._simple_liveness_check(face_pixels)
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            session.close()
            
            logger.info(f"Face verification for {student_id}: match={result['match']}, score={result['similarity']:.4f}")
            
            return result['match'], result['similarity'], liveness_passed, None
            
        except Exception as e:
            logger.error(f"Error in face verification: {e}")
            return False, 0.0, False, str(e)
    
    def _simple_liveness_check(self, face_pixels: np.ndarray) -> bool:
        """
        Simple liveness detection (placeholder implementation)
        You can enhance this with proper liveness detection algorithms
        """
        try:
            # Basic checks for image quality and realness
            # Check if image has sufficient variance (not too dark/bright)
            variance = np.var(face_pixels)
            if variance < 0.01:  # Too uniform
                return False
            
            # Check for reasonable pixel value distribution
            mean_pixel = np.mean(face_pixels)
            if mean_pixel < -0.8 or mean_pixel > 0.8:  # Too dark or too bright
                return False
            
            # For now, assume basic quality checks pass liveness
            return True
            
        except Exception as e:
            logger.error(f"Error in liveness check: {e}")
            return False

# Global ML interface instance
ml_interface = MLInterface()

# Wrapper functions for backward compatibility
def enroll_student(student_id: str, image_files: List) -> Tuple[bool, Optional[str]]:
    """Wrapper function for student enrollment"""
    return ml_interface.enroll_student(student_id, image_files)

def verify_face(student_id: str, selfie_file: Union[BytesIO, object]) -> Tuple[bool, float, bool, Optional[str]]:
    """Wrapper function for face verification"""
    return ml_interface.verify_face(student_id, selfie_file)