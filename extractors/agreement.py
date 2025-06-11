import os
import json
from utils.vertex_ai import process_document
import config

def get_extraction_prompt():
    """Return the prompt for Agreement document extraction."""
    return """
    Extract the following information from the Agreement document with high accuracy. 
    Focus on capturing key details and output them in a structured JSON object format:
    
    {
        "dpn": {
            "borrowersSignatures": "Whether borrowers' signatures are present on the revenue stamp (true/false)",
            "leadID": "The unique lead ID number",
            "customerName": "Full name of the customer/borrower",
            "loanAmount": "The loan amount (numeric value only)"
        },
        "schedulePage": {
            "borrowersSignature": "Whether borrowers' signatures are present (true/false)",
            "cholaAuthorizedSignature": "Whether Chola authorized signature is present (true/false)"
        }
    }
    
    Ensure the JSON object format is clean, with each extracted field labeled precisely by the above field names.
    If a field is not found, return it as an empty string or null to maintain consistency.
    
    Look carefully at all parts of the document including headers, tables, and footnotes.
    Pay special attention to signatures and stamps.
    
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
        "dpn": {
            "borrowersSignatures": structured_data.get("dpn", {}).get("borrowersSignatures", False),
            "leadID": structured_data.get("dpn", {}).get("leadID", ""),
            "customerName": structured_data.get("dpn", {}).get("customerName", ""),
            "loanAmount": structured_data.get("dpn", {}).get("loanAmount", "")
        },
        "schedulePage": {
            "borrowersSignature": structured_data.get("schedulePage", {}).get("borrowersSignature", False),
            "cholaAuthorizedSignature": structured_data.get("schedulePage", {}).get("cholaAuthorizedSignature", False)
        }
    }
    
    return extracted_fields

def extract_details(case_id, file_path):
    """
    Extract details from an Agreement document
    
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
        "document_type": "agreement",
        "file_path": file_path,
        "extracted_data": extracted_fields,
        "raw_response": response.get("raw_response", "")
    }