import os
import json
from utils.vertex_ai import process_document
import config

def get_extraction_prompt():
    """Return the prompt for Sanction Letter extraction."""
    return """
    Extract the following information from the Sanction Letter document with high accuracy. 
    Focus on capturing key details and output them in a structured JSON object format:
    
    {
        "customerName": "Full name of the customer/borrower",
        "loanAmount": "The sanctioned loan amount (numeric value only)",
        "propertyAddress": "Complete address of the property",
        "leadID": "The unique lead ID number",
        "propertyOwnerName": "Name of the property owner",
        "emiAmount": "The EMI amount (numeric value only)",
        "tenure": "Loan tenure in months (numeric value only)",
        "ROI": "Rate of interest percentage (numeric value only)",
        "borrowersSignature": "Whether borrower's signature is present (true/false)",
        "authorizedSignature": "Whether authorized signature is present (true/false)"
    }
    
    Ensure the JSON object format is clean, with each extracted field labeled precisely by the above field names.
    If a field is not found, return it as an empty string or null to maintain consistency.
    
    Look carefully at all parts of the document including headers, tables, and footnotes.
    For numeric values, extract only the numbers without currency symbols or text.
    
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
        "customerName": structured_data.get("customerName", ""),
        "loanAmount": structured_data.get("loanAmount", ""),
        "propertyAddress": structured_data.get("propertyAddress", ""),
        "leadID": structured_data.get("leadID", ""),
        "propertyOwnerName": structured_data.get("propertyOwnerName", ""),
        "emiAmount": structured_data.get("emiAmount", ""),
        "tenure": structured_data.get("tenure", ""),
        "ROI": structured_data.get("ROI", ""),
        "borrowersSignature": structured_data.get("borrowersSignature", False),
        "authorizedSignature": structured_data.get("authorizedSignature", False)
    }
    
    return extracted_fields

def extract_details(case_id, file_path):
    """
    Extract details from a Sanction Letter document
    
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
        "document_type": "sanction_letter",
        "file_path": file_path,
        "extracted_data": extracted_fields,
        "raw_response": response.get("raw_response", "")
    }