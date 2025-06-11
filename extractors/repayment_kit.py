import os
import json
from utils.vertex_ai import process_document
import config

def get_extraction_prompt():
    """Return the prompt for Repayment Kit extraction."""
    return """
    Extract the following information from the Repayment Kit document with high accuracy. 
    Focus on capturing key details and output them in a structured JSON object format:
    
    {
        "accountHolderName": "Full name of the account holder",
        "accountNumber": "Complete bank account number",
        "ifscCode": "IFSC code of the bank",
        "accountType": "Type of account (Savings/Current)",
        "customerSignature": "Whether customer signature is present (true/false)",
        "inFavour": "Name of the entity in whose favor the repayment is set up",
        "enachSpdc": "Details about ENACH/SPDC setup"
    }
    
    Ensure the JSON object format is clean, with each extracted field labeled precisely by the above field names.
    If a field is not found, return it as an empty string or null to maintain consistency.
    
    Look carefully at all parts of the document including headers, tables, and footnotes.
    Pay special attention to bank details and mandate information.
    
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
        "accountHolderName": structured_data.get("accountHolderName", ""),
        "accountNumber": structured_data.get("accountNumber", ""),
        "ifscCode": structured_data.get("ifscCode", ""),
        "accountType": structured_data.get("accountType", ""),
        "customerSignature": structured_data.get("customerSignature", False),
        "inFavour": structured_data.get("inFavour", ""),
        "enachSpdc": structured_data.get("enachSpdc", "")
    }
    
    return extracted_fields

def extract_details(case_id, file_path):
    """
    Extract details from a Repayment Kit document
    
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
        "document_type": "repayment_kit",
        "file_path": file_path,
        "extracted_data": extracted_fields,
        "raw_response": response.get("raw_response", "")
    }