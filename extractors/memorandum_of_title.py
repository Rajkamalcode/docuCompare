import os
import json
from utils.vertex_ai import process_document
import config

def get_extraction_prompt():
    """Return the prompt for Memorandum of Title document extraction."""
    return """
    Extract the following information from the Memorandum of Title document with high accuracy. 
    Focus on capturing key details and output them in a structured JSON object format:
    
    {
        "customerName": "Full name of the customer/borrower",
        "loanAmount": "The loan amount (numeric value only)",
        "fourBoundaries": "The four boundaries of the property (North, South, East, West)",
        "propertyAddress": "Complete address of the property",
        "inFavour": "Name of the entity in whose favor the memorandum is created"
    }
    
    Ensure the JSON object format is clean, with each extracted field labeled precisely by the above field names.
    If a field is not found, return it as an empty string or null to maintain consistency.
    
    Look carefully at all parts of the document including headers, tables, and footnotes.
    For the boundaries, capture the complete description of what exists on each side of the property.
    Check if "Cholamandalam Investment and finance company limited" is mentioned as the entity in favor.
    
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
        "fourBoundaries": structured_data.get("fourBoundaries", ""),
        "propertyAddress": structured_data.get("propertyAddress", ""),
        "inFavour": structured_data.get("inFavour", "")
    }
    
    return extracted_fields

def extract_details(case_id, file_path):
    """
    Extract details from a Memorandum of Title document
    
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
        "document_type": "memorandum_of_title",
        "file_path": file_path,
        "extracted_data": extracted_fields,
        "raw_response": response.get("raw_response", "")
    }