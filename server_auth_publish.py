import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import json
import logging

# --- Setup and Environment ---

# Load environment variables from .env file
load_dotenv()

# --- Configuration (from .env) ---
EXPECTED_API_KEY = os.getenv("UPLOAD_API_KEY")
PUBLISH_DIR_STR = os.getenv("PUBLISH_DIR")

# Input validation
if not EXPECTED_API_KEY or not PUBLISH_DIR_STR:
    print("❌ ERROR: UPLOAD_API_KEY or PUBLISH_DIR not configured in .env. Exiting.")
    sys.exit(1)

PUBLISH_DIR = Path(PUBLISH_DIR_STR)

# Ensure the publish directory exists on the server
if not PUBLISH_DIR.is_dir():
    print(f"❌ ERROR: Publish directory not found or is not a directory: {PUBLISH_DIR_STR}")
    sys.exit(1)

# Configure Flask
app = Flask(__name__)

# 🟢 CRITICAL RAM LIMIT: Set max content length to 16MB (16 * 1024 * 1024 bytes)
# This prevents excessively large files from crashing the 1GB limit.
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Set up basic logging
# 🟢 OPTIMIZATION: Use WARNING level to reduce I/O overhead from logs
logging.basicConfig(level=logging.WARNING) 
logger = logging.getLogger(__name__)


# --- Core Functions ---

def authenticate_request():
    """Authenticates the incoming request using the Bearer token."""
    auth_header = request.headers.get('Authorization')
    
    if not auth_header:
        logger.warning("Auth attempt failed: No Authorization header.")
        return False
        
    try:
        scheme, token = auth_header.split()
    except ValueError:
        logger.warning("Auth attempt failed: Invalid Authorization header format.")
        return False
        
    if scheme.lower() != 'bearer':
        logger.warning(f"Auth attempt failed: Invalid scheme '{scheme}'.")
        return False
        
    if token != EXPECTED_API_KEY:
        logger.error(f"Auth attempt failed: Key mismatch. Received token starts with '{token[:10]}...'")
        return False
        
    logger.warning("✅ Authentication successful.")
    return True

# --- API Endpoint ---

@app.route('/insights-upload', methods=['POST'])
def insights_upload():
    """
    Receives files from the local script, authenticates, and publishes them.
    """
    if not authenticate_request():
        return jsonify({"message": "Authentication Failed. Access Denied."}), 401
    
    if 'html_file' not in request.files or 'image_file' not in request.files or 'metadata_file' not in request.files:
        return jsonify({
            "message": "Missing one or more required files (html_file, image_file, metadata_file)."
        }), 400

    html_file = request.files['html_file']
    image_file = request.files['image_file']
    metadata_file = request.files['metadata_file']
    
    logger.warning(f"Receiving files: {html_file.filename}, {image_file.filename}, {metadata_file.filename}")

    published_paths = []
    
    try:
        # A. Save the HTML file to the public directory
        html_path = PUBLISH_DIR / html_file.filename
        html_file.save(html_path)
        published_paths.append(html_path)
        logger.warning(f"Published HTML: {html_path.name}")
        
        # B. Save the Image file (JPG/SVG placeholder)
        image_path = PUBLISH_DIR / image_file.filename
        image_file.save(image_path)
        published_paths.append(image_path)
        logger.warning(f"Published Image: {image_path.name}")
        
        # C. Process the Metadata file
        # Reading and parsing a small JSON file is very low-memory
        metadata_content = metadata_file.read()
        metadata_json = json.loads(metadata_content)
        
        # Save the JSON file itself 
        metadata_path = PUBLISH_DIR / metadata_file.filename
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_json, f, indent=4)
        published_paths.append(metadata_path)
        logger.warning(f"Metadata JSON saved: {metadata_path.name}")
        
        
        # Post-Processing Step (Indexing)
        logger.warning("Post-processing step completed (Check indexing script).")

        return jsonify({
            "message": "Files received and published successfully.",
            "paths": [p.name for p in published_paths],
            "insight_title": metadata_json.get('title')
        }), 200

    except Exception as e:
        logger.error(f"An internal server error occurred: {e}", exc_info=True)
        return jsonify({
            "message": f"Server processing failed. Error: {str(e)}",
            "paths_saved_before_error": [p.name for p in published_paths]
        }), 500

# --- Run the App ---
if __name__ == '__main__':
    # Use a specific host and port for deployment
    logger.warning("Starting Flask insights API server...")
    app.run(host='0.0.0.0', port=5000)