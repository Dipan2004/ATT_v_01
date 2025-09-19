from flask import Flask
from attendance.db import init_db
from attendance.routes import bp as attendance_bp
import os
import logging

def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['DATABASE_URL'] = 'sqlite:///database/database.db'
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['SYNCED_IMAGES_FOLDER'] = 'synced_images'
    app.config['GEO_RADIUS_METERS'] = 50
    app.config['QR_HMAC_SECRET'] = 'your-secret-key-here'
    
    # Create directories
    os.makedirs('database', exist_ok=True)
    os.makedirs('uploads/enrollment', exist_ok=True)
    os.makedirs('uploads/attendance', exist_ok=True)
    os.makedirs('synced_images', exist_ok=True)
    os.makedirs('synced_images/processed', exist_ok=True)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize database
    init_db(app.config['DATABASE_URL'])
    
    # Register blueprints
    app.register_blueprint(attendance_bp, url_prefix='/api')
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)