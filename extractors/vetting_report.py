import os
import json
from utils.vertex_ai import process_document
import config

def get_extraction_prompt():
    """Return the prompt for Vetting Report extraction."""
    return """
    Extract the following information from the Vetting Report document with high accuracy. 
    Focus on capturing key details and output them in a structured JSON object format:
    
    {
        "date": "Date of the vetting report in DD/MM/YYYY format",
        "customerName": "Full name of the customer/borrower",
        "legalVendorSignature": "Whether legal vendor signature is present (true/false)"
    }
    
    Ensure the JSON object format is clean, with each extracted field labeled precisely by the above field names.
    If a field is not found, return it as an empty string or null to maintain consistency.
    
    Look carefully at all parts of the document including headers, tables, and footnotes.
    Pay special attention to dates and signatures.
    
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
        "date": structured_data.get("date", ""),
        "customerName": structured_data.get("customerName", ""),
        "legalVendorSignature": structured_data.get("legalVendorSignature", False)
    }
    
    return extracted_fields

def extract_details(case_id, file_path):
    """
    Extract details from a Vetting Report document
    
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
        "document_type": "vetting_report",
        "file_path": file_path,
        "extracted_data": extracted_fields,
        "raw_response": response.get("raw_response", "")
    }