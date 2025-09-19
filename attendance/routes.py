from flask import Blueprint, request, jsonify, current_app
from attendance.utils import verify_signature, within_radius, parse_iso_ts, code_within_window
from attendance.db import get_session
from database.models import Broadcast, AttendanceEntry, Student, BatchSync
from ml_module import ml_interface
from datetime import datetime, timezone
import traceback
import os
import base64
import uuid
import json

bp = Blueprint("attendance", __name__)

@bp.route("/broadcast/create", methods=["POST"])
def create_broadcast():
    """Professor creates a broadcast entry"""
    try:
        j = request.get_json(force=True)
        prof = j.get("professor_id")
        code = j.get("code")
        vf = parse_iso_ts(j.get("valid_from"))
        vt = parse_iso_ts(j.get("valid_to"))
        lat = j.get("classroom_lat")
        lon = j.get("classroom_lon")
        sig = j.get("signature")

        if not (prof and code and vf and vt and sig):
            return jsonify({"status":"error", "code":"ERR_MISSING_FIELDS"}), 400

        session = get_session()
        broadcast = Broadcast(
            professor_id=prof, code=code,
            classroom_lat=lat, classroom_lon=lon,
            valid_from=vf, valid_to=vt, signature=sig
        )
        session.add(broadcast)
        session.commit()
        broadcast_id = broadcast.id
        session.close()
        
        return jsonify({"status":"ok","broadcast_id": broadcast_id}), 201
        
    except Exception as e:
        current_app.logger.error("create_broadcast error: %s", traceback.format_exc())
        return jsonify({"status":"error","code":"ERR_SERVER","msg": str(e)}), 500

@bp.route("/student/enroll", methods=["POST"])
def student_enroll():
    """Student enrollment endpoint"""
    try:
        student_id = request.form.get("student_id")
        rollno = request.form.get("rollno")
        name = request.form.get("name")
        email = request.form.get("email")
        files = request.files.getlist("images")
        
        if not student_id:
            return jsonify({"status":"error","code":"ERR_MISSING_STUDENT_ID"}), 400

        session = get_session()
        
        # Create or update student
        student = session.query(Student).filter_by(student_id=student_id).first()
        if not student:
            student = Student(
                student_id=student_id, 
                rollno=rollno,
                name=name, 
                email=email
            )
            session.add(student)
        else:
            student.rollno = rollno or student.rollno
            student.name = name or student.name
            student.email = email or student.email

        session.commit()
        session.close()

        # Process enrollment through ML module
        success, error = ml_interface.enroll_student(student_id, files)
        if not success:
            return jsonify({"status":"error","code":"ERR_ML_ENROLL","msg":error}), 500

        return jsonify({"status":"ok", "student_id": student_id}), 200
        
    except Exception as e:
        current_app.logger.error("student_enroll error: %s", traceback.format_exc())
        return jsonify({"status":"error","code":"ERR_SERVER","msg": str(e)}), 500

@bp.route("/sync_images", methods=["POST"])
def sync_images():
    """
    Handle batch image sync from mobile app
    Expected format: {"entries": [{"uniqueKey": "user123|20.36|85.81|2305665", "img_data": "base64..."}]}
    """
    try:
        data = request.get_json()
        if not data or "entries" not in data:
            return jsonify({"status": "error", "message": "Missing entries"}), 400

        entries = data["entries"]
        if not isinstance(entries, list) or len(entries) == 0:
            return jsonify({"status": "error", "message": "Entries must be a non-empty list"}), 400

        # Generate batch ID for this sync
        batch_id = str(uuid.uuid4())
        saved_keys = []
        failed_keys = []
        
        session = get_session()
        
        # Create batch sync record
        batch_sync = BatchSync(
            batch_id=batch_id,
            total_entries=len(entries),
            status='PROCESSING'
        )
        session.add(batch_sync)
        session.commit()

        synced_folder = current_app.config.get('SYNCED_IMAGES_FOLDER', 'synced_images')
        
        for entry in entries:
            try:
                unique_key = entry.get("uniqueKey")
                img_data = entry.get("img_data")

                if not unique_key or not img_data:
                    failed_keys.append(unique_key or "unknown")
                    continue

                # Parse unique key: "user123|20.3623099|85.8159783|2305665"
                parts = unique_key.split("|")
                if len(parts) != 4:
                    failed_keys.append(unique_key)
                    continue
                    
                user_id, lat_str, lon_str, rollno = parts
                
                try:
                    lat = float(lat_str)
                    lon = float(lon_str)
                except ValueError:
                    failed_keys.append(unique_key)
                    continue

                # Decode base64 image
                try:
                    image_bytes = base64.b64decode(img_data)
                except Exception:
                    failed_keys.append(unique_key)
                    continue

                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
                filename = f"{rollno}_{timestamp}.jpg"
                filepath = os.path.join(synced_folder, filename)

                # Ensure directory exists
                os.makedirs(os.path.dirname(filepath), exist_ok=True)

                # Save image file
                with open(filepath, "wb") as f:
                    f.write(image_bytes)

                # Create attendance entry record
                attendance_entry = AttendanceEntry(
                    user_id=user_id,
                    rollno=rollno,
                    student_lat=lat,
                    student_lon=lon,
                    image_path=filepath,
                    batch_id=batch_id,
                    timestamp=datetime.now(timezone.utc),
                    verification_status='PENDING'
                )
                session.add(attendance_entry)
                saved_keys.append(unique_key)

            except Exception as e:
                current_app.logger.error(f"Error processing entry {unique_key}: {e}")
                failed_keys.append(unique_key)

        # Update batch sync record
        batch_sync.processed_entries = len(saved_keys)
        batch_sync.status = 'COMPLETED' if len(failed_keys) == 0 else 'PARTIAL'
        batch_sync.completed_at = datetime.now(timezone.utc)
        
        session.commit()
        session.close()

        # Start background processing of faces (you can make this async)
        try:
            _process_batch_faces(batch_id)
        except Exception as e:
            current_app.logger.error(f"Error starting face processing for batch {batch_id}: {e}")

        return jsonify({
            "status": "success",
            "saved_count": len(saved_keys),
            "failed_count": len(failed_keys),
            "saved_keys": saved_keys,
            "batch_id": batch_id
        }), 200

    except Exception as e:
        current_app.logger.error(f"Sync error: {e}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500

@bp.route("/batch_status/<batch_id>", methods=["GET"])
def get_batch_status(batch_id):
    """Get the processing status of a batch"""
    try:
        session = get_session()
        batch_sync = session.query(BatchSync).filter_by(batch_id=batch_id).first()
        
        if not batch_sync:
            session.close()
            return jsonify({"status": "error", "message": "Batch not found"}), 404
        
        # Get attendance entries for this batch
        entries = session.query(AttendanceEntry).filter_by(batch_id=batch_id).all()
        
        verification_stats = {
            "pending": sum(1 for e in entries if e.verification_status == 'PENDING'),
            "verified": sum(1 for e in entries if e.verification_status == 'VERIFIED'),
            "rejected": sum(1 for e in entries if e.verification_status == 'REJECTED')
        }
        
        result = {
            "batch_id": batch_id,
            "status": batch_sync.status,
            "total_entries": batch_sync.total_entries,
            "processed_entries": batch_sync.processed_entries,
            "successful_verifications": batch_sync.successful_verifications,
            "failed_verifications": batch_sync.failed_verifications,
            "verification_stats": verification_stats,
            "created_at": batch_sync.created_at.isoformat() if batch_sync.created_at else None,
            "completed_at": batch_sync.completed_at.isoformat() if batch_sync.completed_at else None
        }
        
        session.close()
        return jsonify(result), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting batch status: {e}")
        return jsonify({"status": "error", "message": "Internal server error"}), 500

def _process_batch_faces(batch_id: str):
    """
    Process faces for a batch of attendance entries
    This can be made asynchronous with Celery or similar
    """
    try:
        session = get_session()
        
        # Get all pending entries for this batch
        entries = session.query(AttendanceEntry).filter_by(
            batch_id=batch_id, 
            verification_status='PENDING'
        ).all()
        
        successful_verifications = 0
        failed_verifications = 0
        
        for entry in entries:
            try:
                # Find student by rollno (assuming rollno maps to student_id)
                student = session.query(Student).filter_by(rollno=entry.rollno).first()
                if not student or not student.is_enrolled:
                    entry.verification_status = 'REJECTED'
                    entry.processed_at = datetime.now(timezone.utc)
                    failed_verifications += 1
                    continue
                
                # Load image as BytesIO for ML processing
                if not os.path.exists(entry.image_path):
                    entry.verification_status = 'REJECTED'
                    entry.processed_at = datetime.now(timezone.utc)
                    failed_verifications += 1
                    continue
                
                # Create BytesIO from saved image
                with open(entry.image_path, 'rb') as f:
                    image_bytes = f.read()
                
                from io import BytesIO
                image_bio = BytesIO(image_bytes)
                
                # Perform face verification
                face_match, confidence_score, liveness_passed, error = ml_interface.verify_face(
                    student.student_id, image_bio
                )
                
                # Update entry with results
                entry.face_match_score = confidence_score
                entry.is_face_matched = face_match
                entry.liveness_passed = liveness_passed
                entry.processed_at = datetime.now(timezone.utc)
                
                if error:
                    entry.verification_status = 'REJECTED'
                    failed_verifications += 1
                elif face_match and liveness_passed:
                    entry.verification_status = 'VERIFIED'
                    successful_verifications += 1
                else:
                    entry.verification_status = 'REJECTED'
                    failed_verifications += 1
                    
            except Exception as e:
                current_app.logger.error(f"Error processing entry {entry.id}: {e}")
                entry.verification_status = 'REJECTED'
                entry.processed_at = datetime.now(timezone.utc)
                failed_verifications += 1
        
        # Update batch sync statistics
        batch_sync = session.query(BatchSync).filter_by(batch_id=batch_id).first()
        if batch_sync:
            batch_sync.successful_verifications = successful_verifications
            batch_sync.failed_verifications = failed_verifications
            batch_sync.status = 'COMPLETED'
            batch_sync.completed_at = datetime.now(timezone.utc)
        
        session.commit()
        session.close()
        
        current_app.logger.info(f"Batch {batch_id} processing completed: {successful_verifications} verified, {failed_verifications} rejected")
        
    except Exception as e:
        current_app.logger.error(f"Error processing batch {batch_id}: {e}")