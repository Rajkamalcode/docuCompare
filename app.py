from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import json
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

# Import our modules
from utils.db import DocumentDB
from extractors import extract_document
from utils.comparison import compare_documents
import config

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.DOCUMENTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Initialize database
db = DocumentDB()

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Helper functions
def allowed_file(filename):
    """Check if file has an allowed extension"""
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'doc', 'docx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Upload a document file
    
    Returns:
        JSON with file path
    """
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
            
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        if file and allowed_file(file.filename):
            # Generate a unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
            
            # Save the file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Return the file path
            return jsonify({
                "status": "success",
                "file_path": file_path,
                "relative_path": f"documents/{unique_filename}"
            })
        else:
            return jsonify({"error": "File type not allowed"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/process_document', methods=['POST'])
def process_document():
    """
    Process a document and extract information
    
    Expected payload:
    {
        "case_id": "unique_case_id",
        "document_type": "sanction_letter",
        "file_path": "path/to/document.pdf"
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        if not all(k in data for k in ['case_id', 'document_type', 'file_path']):
            return jsonify({"error": "Missing required fields"}), 400
        
        case_id = data['case_id']
        document_type = data['document_type']
        file_path = data['file_path']
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({"error": f"File not found: {file_path}"}), 404
        
        # Extract document details
        result = extract_document(case_id, document_type, file_path)
        
        # Store in database
        db.store_document_data(
            case_id=case_id,
            document_type=document_type,
            extracted_data=result['extracted_data'],
            file_path=file_path
        )
        
        return jsonify({
            "status": "success",
            "case_id": case_id,
            "document_type": document_type,
            "extracted_data": result['extracted_data']
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/get_document', methods=['GET'])
def get_document():
    """
    Get extracted document data
    
    Query parameters:
    - case_id: Unique case ID
    - document_type: (Optional) Type of document to retrieve
    """
    try:
        case_id = request.args.get('case_id')
        document_type = request.args.get('document_type')
        
        if not case_id:
            return jsonify({"error": "Missing case_id parameter"}), 400
        
        # Retrieve document data
        result = db.get_document_data(case_id, document_type)
        
        if not result:
            return jsonify({"error": "Document not found"}), 404
        
        return jsonify({"status": "success", "data": result})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/compare_documents', methods=['POST'])
def compare_documents_api():
    """
    Compare documents for a case
    
    Expected payload:
    {
        "case_id": "unique_case_id",
        "documents": [
            {
                "document_type": "sanction_letter",
                "extracted_data": {...}
            },
            {
                "document_type": "legal_report",
                "extracted_data": {...}
            },
            ...
        ]
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        if not all(k in data for k in ['case_id', 'documents']):
            return jsonify({"error": "Missing required fields"}), 400
        
        case_id = data['case_id']
        documents = data['documents']
        
        # Organize documents by type
        documents_by_type = {}
        for doc in documents:
            if 'document_type' in doc and 'extracted_data' in doc:
                documents_by_type[doc['document_type']] = doc
        
        # Compare documents
        comparison_results = compare_documents(case_id, documents_by_type)
        
        # Store comparison results
        db.store_comparison_results(case_id, comparison_results)
        
        return jsonify({
            "status": "success",
            "case_id": case_id,
            "comparison_results": comparison_results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/get_comparison', methods=['GET'])
def get_comparison():
    """
    Get comparison results for a case
    
    Query parameters:
    - case_id: Unique case ID
    """
    try:
        case_id = request.args.get('case_id')
        
        if not case_id:
            return jsonify({"error": "Missing case_id parameter"}), 400
        
        # Retrieve comparison results
        result = db.get_comparison_results(case_id)
        
        if not result:
            # If no stored comparison results, generate them on the fly
            documents = db.get_document_data(case_id)
            
            if not documents:
                return jsonify({"error": "No documents found for this case"}), 404
            
            # Organize documents by type
            documents_by_type = {}
            for doc in documents:
                documents_by_type[doc['document_type']] = doc
            
            # Compare documents
            comparison_results = compare_documents(case_id, documents_by_type)
            
            # Store comparison results
            db.store_comparison_results(case_id, comparison_results)
            
            return jsonify({
                "status": "success",
                "case_id": case_id,
                "comparison_results": comparison_results,
                "generated": "on-demand"
            })
        
        return jsonify({
            "status": "success",
            "case_id": case_id,
            "comparison_results": result['comparison_data']
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/get_cases', methods=['GET'])
def get_cases():
    """Get a list of all cases"""
    try:
        cases = db.get_all_cases()
        return jsonify({"status": "success", "cases": cases})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/documents/<path:filename>')
def serve_document(filename):
    """Serve document files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/process_all', methods=['POST'])
def process_all_documents():
    """
    Process multiple documents for a case in a single request
    
    Expected payload:
    {
        "case_id": "unique_case_id",
        "documents": [
            {
                "document_type": "sanction_letter",
                "file_path": "path/to/document.pdf"
            },
            {
                "document_type": "legal_report",
                "file_path": "path/to/document.pdf"
            },
            ...
        ]
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        if not all(k in data for k in ['case_id', 'documents']):
            return jsonify({"error": "Missing required fields"}), 400
        
        case_id = data['case_id']
        documents = data['documents']
        
        results = []
        documents_by_type = {}
        
        # Process each document
        for doc in documents:
            if not all(k in doc for k in ['document_type', 'file_path']):
                return jsonify({"error": "Missing document fields"}), 400
            
            document_type = doc['document_type']
            file_path = doc['file_path']
            
            # Check if file exists
            if not os.path.exists(file_path):
                return jsonify({"error": f"File not found: {file_path}"}), 404
            
            # Extract document details
            result = extract_document(case_id, document_type, file_path)
            
            # Store in database
            db.store_document_data(
                case_id=case_id,
                document_type=document_type,
                extracted_data=result['extracted_data'],
                file_path=file_path
            )
            
            # Add to results
            results.append({
                "document_type": document_type,
                "extracted_data": result['extracted_data']
            })
            
            # Add to documents by type for comparison
            documents_by_type[document_type] = {
                "document_type": document_type,
                "extracted_data": result['extracted_data'],
                "file_path": file_path
            }
        
        # Compare documents
        comparison_results = compare_documents(case_id, documents_by_type)
        
        # Store comparison results
        db.store_comparison_results(case_id, comparison_results)
        
        return jsonify({
            "status": "success",
            "case_id": case_id,
            "processed_documents": results,
            "comparison_results": comparison_results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5444)
