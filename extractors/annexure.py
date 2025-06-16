import os
os.environ['USE_TF'] = '0'  # Force DocTR to use PyTorch
import torch
import json
import logging
from utils.vertex_ai import process_document
from utils.ollama import call_ollama_api
import config
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_extraction_prompt():
    """Return the prompt for Annexure document extraction."""
    return """
    Extract the following information from the Annexure document with high accuracy. 
    Focus on capturing key details and output them in a structured JSON object format:
    
    {
        "date": "Date of the annexure in DD/MM/YYYY format",
        "leadID": "The unique lead ID number",
        "branch": "Branch name or code",
        "customerName": "Full name of the customer/borrower usually will be in the consent form starting with I am ...",
        "authorizedSignature": "Whether authorized signature is present (true/false)"
    }
    
    Ensure the JSON object format is clean, with each extracted field labeled precisely by the above field names.
    If a field is not found, return it as an empty string or null to maintain consistency.
    
    Look carefully at all parts of the document including headers, tables, and footnotes.
    Pay special attention to dates, IDs, and signatures.
    
    Return ONLY the JSON object without any additional text, explanations, or markdown formatting.
    """

def get_validation_prompt(extracted_json):
    """Return a prompt to validate the extracted JSON."""
    return f"""
    Review the following JSON extracted from an Annexure document:
    
    {json.dumps(extracted_json, indent=2)}
    
    Evaluate the quality and completeness of the extraction. Check for:
    1. Missing critical fields (especially date, leadID, customerName)
    2. Obvious errors in extraction (wrong formats, nonsensical values)
    3. Incomplete extractions where more information is likely available
    
    Return a JSON with this format:
    {{
        "is_valid": true/false,
        "confidence_score": 0-100,
        "missing_fields": ["field1", "field2"],
        "error_fields": ["field3", "field4"],
        "recommendation": "proceed" or "fallback"
    }}
    
    Return ONLY the JSON object without any additional text.
    """

def extract_text_with_doctr(file_path):
    """
    Extract text from document using docTR OCR
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Extracted text as a string
    """
    try:
        # Load the document
        doc = DocumentFile.from_pdf(file_path)
        
        # Load the OCR predictor
        predictor = ocr_predictor(pretrained=True)
        
        # Analyze the document
        result = predictor(doc)
        
        # Extract text
        extracted_text = result.export()
        
        # Convert the structured result to a plain text string
        full_text = ""
        for page in extracted_text["pages"]:
            for block in page["blocks"]:
                for line in block["lines"]:
                    for word in line["words"]:
                        full_text += word["value"] + " "
                    full_text += "\n"
                full_text += "\n"
        
        return full_text
    except Exception as e:
        logger.error(f"Error in docTR OCR processing: {e}")
        return None

def extract_fields_with_ollama(text):
    """
    Extract fields from document text using Ollama
    
    Args:
        text: Document text
        
    Returns:
        Dictionary with extracted fields and validation result
    """
    # Get the extraction prompt
    prompt = get_extraction_prompt() + "\n\nDocument text:\n" + text
    
    # Call Ollama API for extraction
    response = call_ollama_api(prompt, model_name="gemma3:12b-it-qat", step_name="Extraction")
    
    if not response:
        logger.error("Ollama extraction failed")
        return None, None
    
    # Try to parse the response as JSON
    try:
        extracted_fields = json.loads(response)
        
        # Validate the extraction
        validation_prompt = get_validation_prompt(extracted_fields)
        validation_response = call_ollama_api(validation_prompt, model_name="gemma3:12b-it-qat", step_name="Validation")
        
        if validation_response:
            try:
                validation_result = json.loads(validation_response)
                return extracted_fields, validation_result
            except json.JSONDecodeError:
                logger.error(f"Failed to parse validation response as JSON: {validation_response[:200]}")
                return extracted_fields, None
        else:
            logger.error("Ollama validation failed")
            return extracted_fields, None
            
    except json.JSONDecodeError:
        logger.error(f"Failed to parse Ollama response as JSON: {response[:200]}")
        return None, None

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
        "leadID": structured_data.get("leadID", ""),
        "branch": structured_data.get("branch", ""),
        "customerName": structured_data.get("customerName", ""),
        "authorizedSignature": structured_data.get("authorizedSignature", False)
    }
    
    return extracted_fields

def extract_details(case_id, file_path):
    """
    Extract details from an Annexure document
    
    Args:
        case_id: Unique identifier for the document case
        file_path: Path to the document file
        
    Returns:
        Dictionary with extracted fields
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Extract text using docTR OCR
    document_text = extract_text_with_doctr(file_path)
    
    if not document_text:
        logger.error("OCR extraction failed, falling back to Vertex AI")
        use_vertex_fallback = True
    else:
        # Try extraction with Ollama
        extracted_fields, validation_result = extract_fields_with_ollama(document_text)
        
        # Determine if we need to fall back to Vertex AI
        use_vertex_fallback = False
        
        if not extracted_fields:
            logger.info("Ollama extraction failed, falling back to Vertex AI")
            use_vertex_fallback = True
        elif validation_result and validation_result.get("recommendation") == "fallback":
            logger.info(f"Ollama validation recommends fallback to Vertex AI. Score: {validation_result.get('confidence_score')}")
            use_vertex_fallback = True
    
    # Fall back to Vertex AI if needed
    if use_vertex_fallback:
        # Get the extraction prompt
        prompt = get_extraction_prompt()
        
        # Process the document using Vertex AI
        response = process_document(file_path, prompt)
        
        # Extract and structure the fields
        extracted_fields = extract_fields(response)
        
        method_used = "vertex_ai"
        raw_response = response.get("raw_response", "")
        print(f"✅ Extraction completed using Vertex AI")
    else:
        method_used = "ollama"
        raw_response = json.dumps(extracted_fields)
        print(f"✅ Extraction completed using Ollama")
    
    return {
        "case_id": case_id,
        "document_type": "annexure",
        "file_path": file_path,
        "extracted_data": extracted_fields,
        "method_used": method_used,
        "raw_response": raw_response,
        "validation_result": validation_result if method_used == "ollama" and validation_result else None
    }

def main():
    """Test function for annexure extraction"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract details from an Annexure document')
    parser.add_argument('file_path', help='Path to the document file')
    parser.add_argument('--case-id', default='test_case', help='Unique identifier for the document case')
    
    args = parser.parse_args()
    
    try:
        result = extract_details(args.case_id, args.file_path)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
