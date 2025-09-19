from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime, timezone

Base = declarative_base()

class Student(Base):
    __tablename__ = "students"
    
    id = Column(Integer, primary_key=True)
    student_id = Column(String(64), unique=True, nullable=False)
    rollno = Column(String(64), nullable=True)
    name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True)
    face_embedding = Column(Text, nullable=True)  # JSON serialized embedding
    enrolled_photo_path = Column(String(512), nullable=True)
    is_enrolled = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc))

class Broadcast(Base):
    __tablename__ = "broadcasts"
    
    id = Column(Integer, primary_key=True)
    professor_id = Column(String(128), nullable=False)
    code = Column(String(256), nullable=False, index=True)
    classroom_lat = Column(Float, nullable=True)
    classroom_lon = Column(Float, nullable=True)
    valid_from = Column(DateTime, nullable=False)
    valid_to = Column(DateTime, nullable=False)
    signature = Column(String(512), nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

class AttendanceEntry(Base):
    __tablename__ = "attendance_entries"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(64), nullable=True)
    rollno = Column(String(64), nullable=False)
    student_lat = Column(Float, nullable=False)
    student_lon = Column(Float, nullable=False)
    image_path = Column(String(512), nullable=False)
    timestamp = Column(DateTime, default=datetime.now(timezone.utc))
    
    # Face verification results
    face_match_score = Column(Float, nullable=True)
    is_face_matched = Column(Boolean, nullable=True)
    liveness_passed = Column(Boolean, nullable=True)
    verification_status = Column(String(32), default='PENDING')  # PENDING, VERIFIED, REJECTED
    
    # Batch processing info
    batch_id = Column(String(128), nullable=True)
    processed_at = Column(DateTime, nullable=True)
    
    # Optional link to broadcast
    broadcast_id = Column(Integer, ForeignKey("broadcasts.id"), nullable=True)
    broadcast = relationship("Broadcast")

class BatchSync(Base):
    __tablename__ = "batch_syncs"
    
    id = Column(Integer, primary_key=True)
    batch_id = Column(String(128), unique=True, nullable=False)
    total_entries = Column(Integer, default=0)
    processed_entries = Column(Integer, default=0)
    successful_verifications = Column(Integer, default=0)
    failed_verifications = Column(Integer, default=0)
    status = Column(String(32), default='PROCESSING')  # PROCESSING, COMPLETED, FAILED
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)