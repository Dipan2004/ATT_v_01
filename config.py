import os

class Config:
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///database/database.db'
    UPLOAD_FOLDER = 'uploads'
    SYNCED_IMAGES_FOLDER = 'synced_images'
    GEO_RADIUS_METERS = int(os.environ.get('GEO_RADIUS_METERS', '50'))
    QR_HMAC_SECRET = os.environ.get('QR_HMAC_SECRET', 'your-secret-key-here')
    ML_CONFIDENCE_THRESHOLD = float(os.environ.get('ML_CONFIDENCE_THRESHOLD', '0.6'))