import os
import json
from utils.vertex_ai import process_document
import config

def get_extraction_prompt():
    """Return the prompt for KYC document extraction."""
    return """
    Extract the following information from the KYC document with high accuracy. 
    Focus on capturing key details and output them in a structured JSON object format:
    
    {
        "name": "Full name of the person",
        "dob": "Date of birth in DD/MM/YYYY format",
        "gender": "Gender (Male/Female/Other)",
        "address": "Complete residential address",
        "kycNumber": "KYC document number (Aadhaar/PAN/etc.)",
        "aadhaarNumber": "Aadhaar number usually are masked with 8 numbers beign abstracted, take the last 4 numbers."
    }
    
    Ensure the JSON object format is clean, with each extracted field labeled precisely by the above field names.
    If a field is not found, return it as an empty string or null to maintain consistency.
    
    Look carefully at all parts of the document including headers, tables, and footnotes.
    For Aadhaar numbers, check if they are already masked. If not, indicate that the first 8 digits should be masked.
    
    Return ONLY the JSON object without any additional text, explanations, or markdown formatting.
    """

def extract_fields(response_data):
    """
    Extract and structure fields from the Vertex AI response
    
    Args:
        response_data: Response from Vertex AI
        
    Returns:
        Dictionary with structured field data
    """
    if not response_data or "structured_data" not in response_data:
        return {}
    
    structured_data = response_data["structured_data"]
    
    # Map the fields from the response to our expected format
    extracted_fields = {
        "name": structured_data.get("name", ""),
        "dob": structured_data.get("dob", ""),
        "gender": structured_data.get("gender", ""),
        "address": structured_data.get("address", ""),
        "kycNumber": structured_data.get("kycNumber", ""),
        "aadhaarNumber": structured_data.get("aadhaarNumber", "")
    }
    
    return extracted_fields

def extract_details(case_id, file_path):
    """
    Extract details from a KYC document
    
    Args:
        case_id: Unique identifier for the document case
        file_path: Path to the document file
        
    Returns:
        Dictionary with extracted fields
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get the extraction prompt
    prompt = get_extraction_prompt()
    
    # Process the document using Vertex AI
    response = process_document(file_path, prompt)
    
    # Extract and structure the fields
    extracted_fields = extract_fields(response)
    
    return {
        "case_id": case_id,
        "document_type": "kyc",
        "file_path": file_path,
        "extracted_data": extracted_fields,
        "raw_response": response.get("raw_response", "")
    }