from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import json
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
os.environ['USE_TF'] = '0'  # Force DocTR to use PyTorch

# Import our modules
from utils.db import DocumentDB
from extractors import extract_document
from utils.comparison import compare_documents, set_rapid_system_data
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

@app.route('/api/set_rapid_system', methods=['POST'])
def set_rapid_system():
    """
    Set RAPID_SYSTEM data for comparison
    
    Expected payload:
    {
        "type": "KYC",
        "location": "kyc_document.pdf",
        "fields": {
            "name": "Jane Smith",
            "address": "123 Main St, Bangalore, Karnataka",
            "id_number": "ABCDE1234F",
            "date_of_birth": "10/05/1985"
        }
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        if not all(k in data for k in ['type', 'fields']):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Fix: Check if type is a string before calling lower()
        if isinstance(data['type'], str):
            doc_type = data['type'].lower().replace(' ', '_')
        else:
            # Handle the case where type is not a string (e.g., it's a dict)
            return jsonify({"error": "The 'type' field must be a string"}), 400
        
        # Store in global RAPID_SYSTEM data
        rapid_system_data = getattr(app, 'rapid_system_data', {})
        rapid_system_data[doc_type] = data
        app.rapid_system_data = rapid_system_data
        
        # Update comparison module
        set_rapid_system_data(rapid_system_data)
        
        return jsonify({
            "status": "success",
            "message": f"RAPID_SYSTEM data set for {doc_type}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/api/get_rapid_system', methods=['GET'])
def get_rapid_system():
    """Get all RAPID_SYSTEM data"""
    try:
        rapid_system_data = getattr(app, 'rapid_system_data', {})
        return jsonify({"status": "success", "data": rapid_system_data})
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
        "file_path": "path/to/document.pdf",
        "rapid_system_data": {  # Optional
            "fields": {
                "name": "John Doe",
                "address": "456 Oak St, Mumbai, Maharashtra"
            }
        }
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
        
        # Ensure document_type is a string
        if not isinstance(document_type, str):
            return jsonify({"error": f"Document type must be a string, got {type(document_type)}"}), 400
        
        # Check if RAPID_SYSTEM data is provided
        if 'rapid_system_data' in data and data['rapid_system_data']:
            # Store in global RAPID_SYSTEM data
            rapid_system_data = getattr(app, 'rapid_system_data', {})
            rapid_system_data[document_type] = {
                'type': document_type,
                'location': file_path,
                'fields': data['rapid_system_data'].get('fields', {})
            }
            app.rapid_system_data = rapid_system_data
            
            # Update comparison module
            set_rapid_system_data(rapid_system_data)
        
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
        ],
        "rapid_system_data": {  # Optional
            "kyc": {
                "fields": {
                    "name": "John Doe",
                    "address": "456 Oak St, Mumbai, Maharashtra"
                }
            }
        }
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        if not all(k in data for k in ['case_id', 'documents']):
            return jsonify({"error": "Missing required fields"}), 400
        
        case_id = data['case_id']
        documents = data['documents']
        
        # Check if RAPID_SYSTEM data is provided
        if 'rapid_system_data' in data and data['rapid_system_data']:
            # Store in global RAPID_SYSTEM data
            rapid_system_data = getattr(app, 'rapid_system_data', {})
            for doc_type, doc_data in data['rapid_system_data'].items():
                # Ensure doc_type is a string
                if not isinstance(doc_type, str):
                    return jsonify({"error": f"Document type must be a string, got {type(doc_type)}"}), 400
                
                rapid_system_data[doc_type] = {
                    'type': doc_type,
                    'fields': doc_data.get('fields', {})
                }
            app.rapid_system_data = rapid_system_data
            
            # Update comparison module
            set_rapid_system_data(rapid_system_data)
        
        # Organize documents by type
        documents_by_type = {}
        for doc in documents:
            if 'document_type' in doc and 'extracted_data' in doc:
                # Ensure document_type is a string
                if not isinstance(doc['document_type'], str):
                    return jsonify({"error": f"Document type must be a string, got {type(doc['document_type'])}"}), 400
                
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
                "file_path": "path/to/document.pdf",
                "rapid_system_data": {  # Optional
                    "fields": {
                        "name": "John Doe",
                        "address": "456 Oak St, Mumbai, Maharashtra"
                    }
                }
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
            
            # Ensure document_type is a string
            if not isinstance(document_type, str):
                return jsonify({"error": f"Document type must be a string, got {type(document_type)}"}), 400
            
            # Check if RAPID_SYSTEM data is provided
            if 'rapid_system_data' in doc and doc['rapid_system_data']:
                # Store in global RAPID_SYSTEM data
                rapid_system_data = getattr(app, 'rapid_system_data', {})
                rapid_system_data[document_type] = {
                    'type': document_type,
                    'location': file_path,
                    'fields': doc['rapid_system_data'].get('fields', {})
                }
                app.rapid_system_data = rapid_system_data
                
                # Update comparison module
                set_rapid_system_data(rapid_system_data)
            
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
