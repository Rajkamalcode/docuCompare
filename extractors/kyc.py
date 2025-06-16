import os
os.environ['USE_TF'] = '0'  # Force DocTR to use PyTorch
import json
import re
import logging
from utils.vertex_ai import process_document
from utils.ollama import call_ollama_api
import config
from pypdf import PdfReader, PdfWriter
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_extraction_prompt():
    """Return the prompt for KYC document extraction handling multiple entries."""
    return """
    Extract information for ALL KYC documents present in the provided document.
    There might be multiple KYC entries. For each entry, extract the following fields:

    {
        "name": "Full name of the person",
        "dob": "Date of birth in DD/MM/YYYY format",
        "gender": "Gender (Male/Female/Other)",
        "address": "Complete residential address",
        "kycNumber": "KYC document number (PAN/Voter ID/etc.)",
        "aadhaarNumber": "Last 4 visible digits of Aadhaar (ignore masked portions)"
    }

    IMPORTANT:
    1. Return a JSON ARRAY of objects - one object per KYC entry
    2. For Aadhaar numbers:
        - Extract ONLY the last 4 visible digits (ignore 'X', '*', or masked characters)
        - If full number is visible, still extract only last 4 digits
        - Pay special attention to digit recognition (0,1,2,3,4,5,6,7,8,9)
        - Double-check the digits to ensure accuracy
    3. For PAN numbers (kycNumber):
        - Extract 10-character alphanumeric strings (format: ABCDE1234F)
    4. Return ONLY the JSON array without any additional text
    5. Include ALL entries found in the document
    """

def get_validation_prompt(extracted_json):
    """Return a prompt to validate the extracted JSON."""
    return f"""
    Review the following JSON extracted from a KYC document:
    
    {json.dumps(extracted_json, indent=2)}
    
    Evaluate the quality and completeness of the extraction. Check for:
    1. Missing critical fields (especially name, dob, address, kycNumber)
    2. Obvious errors in extraction (wrong formats, nonsensical values)
    3. Incomplete extractions where more information is likely available
    4. Pay special attention to the Aadhaar number digits - ensure they are correctly recognized
    
    Return a JSON with this format:
    {{
        "is_valid": true/false,
        "confidence_score": 0-100,
        "missing_fields": ["field1", "field2"],
        "error_fields": ["field3", "field4"],
        "recommendation": "proceed" or "fallback",
        "digit_confidence": 0-100
    }}
    
    Return ONLY the JSON object without any additional text.
    """

def extract_text_with_doctr(file_path):
    """
    Extract text from document using docTR OCR with enhanced settings for digit recognition
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Extracted text as a string
    """
    try:
        # Load the document
        doc = DocumentFile.from_pdf(file_path)
        
        # Load the OCR predictor with higher confidence threshold for better accuracy
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
    Extract and structure fields from Vertex AI response, handling multiple entries

    Args:
        response_data: Response from Vertex AI

    Returns:
        List of dictionaries with structured field data
    """
    if not response_data or "structured_data" not in response_data:
        logger.warning("No structured_data in response_data or response_data is empty.")
        return []

    structured_data = response_data["structured_data"]
    entries = []

    # Handle single entry response by converting to list
    if isinstance(structured_data, dict):
        entries = [structured_data]
    elif isinstance(structured_data, list):
        entries = structured_data
    else:
        logger.warning(f"structured_data is not a dict or list, but {type(structured_data)}. Value: {structured_data}")
        return []

    extracted_list = []

    for entry in entries:
        if not isinstance(entry, dict):
            logger.warning(f"Found an item in entries that is not a dictionary: {entry}")
            continue # Skip non-dictionary entries

        # Enhanced Aadhaar number extraction with improved digit handling
        aadhaar = entry.get("aadhaarNumber", "")
        if aadhaar:
            # Convert to string if not already
            aadhaar_str = str(aadhaar)
            
            # First try to find a pattern that looks like an Aadhaar number
            # Look for patterns like "xxxx xxxx 1234" or "XXXX XXXX 1234" or "xxxx-xxxx-1234"
            aadhaar_pattern = re.search(r'[xX*\d]{4}[\s-]?[xX*\d]{4}[\s-]?(\d{4})', aadhaar_str)
            
            if aadhaar_pattern:
                # If we found a pattern, use the last group (the last 4 digits)
                last_four = aadhaar_pattern.group(1)
                aadhaar = f"xxxx xxxx {last_four}"
            else:
                # If no pattern found, extract any digits and take the last 4
                digits = re.findall(r'\d', aadhaar_str)
                if len(digits) >= 4:
                    last_four = ''.join(digits[-4:])
                    aadhaar = f"xxxx xxxx {last_four}"
                elif len(digits) > 0:
                    # If less than 4 digits, pad with x
                    last_digits = ''.join(digits)
                    padding = 'x' * (4 - len(last_digits))
                    aadhaar = f"xxxx xxxx {padding}{last_digits}"
                else:
                    aadhaar = ""
        else:
            aadhaar = ""

        # Enhanced PAN number extraction
        kyc_num = entry.get("kycNumber", "")
        if kyc_num:
            # Convert to string if not already
            kyc_str = str(kyc_num)
            
            # Look for PAN pattern: 5 letters + 4 digits + 1 letter
            pan_match = re.search(r'[A-Za-z]{5}\d{4}[A-Za-z]', kyc_str)
            if pan_match:
                kyc_num = pan_match.group(0).upper()  # Convert to uppercase
            # If no match, keep as is - could be other ID types
        else:
            kyc_num = ""

        extracted_list.append({
            "name": entry.get("name", ""),
            "dob": entry.get("dob", ""),
            "gender": entry.get("gender", ""),
            "address": entry.get("address", ""),
            "kycNumber": kyc_num,
            "aadhaarNumber": aadhaar
        })

    return extracted_list

def extract_details_from_all_pages(case_id, file_path, method="vertex_ai"):
    """
    Extract details from ALL pages of a KYC document.
    This version iterates through pages if `process_document` can't handle multi-page.

    Args:
        case_id: Unique identifier for the document case
        file_path: Path to the document file
        method: Extraction method to use ("vertex_ai" or "ollama")

    Returns:
        Dictionary with list of extracted entries
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    all_extracted_data = []
    all_raw_responses = []
    validation_results = []

    try:
        pdf_reader = PdfReader(file_path)
        num_pages = len(pdf_reader.pages)

        if num_pages == 0:
            logger.warning(f"PDF file {file_path} has 0 pages.")
            # Fallback: try processing the whole file directly if pypdf fails to read pages
            logger.info(f"Attempting to process file {file_path} as a whole...")
            
            if method == "ollama":
                document_text = extract_text_with_doctr(file_path)
                if document_text:
                    extracted_fields, validation_result = extract_fields_with_ollama(document_text)
                    if extracted_fields:
                        all_extracted_data.extend(extracted_fields)
                        all_raw_responses.append(json.dumps(extracted_fields))
                        if validation_result:
                            validation_results.append(validation_result)
                    else:
                        all_raw_responses.append("No response from Ollama for whole file.")
                else:
                    all_raw_responses.append("DocTR failed to extract text from whole file.")
            else:  # vertex_ai
                response = process_document(file_path, get_extraction_prompt())
                extracted_page_data = extract_fields(response)
                all_extracted_data.extend(extracted_page_data)
                if response:
                    all_raw_responses.append(response.get("raw_response", ""))
                else:
                    all_raw_responses.append("No response from process_document for whole file.")

        for page_num in range(num_pages):
            pdf_writer = PdfWriter()
            pdf_writer.add_page(pdf_reader.pages[page_num])

            temp_page_path = f"{os.path.splitext(file_path)[0]}_page_{page_num + 1}.pdf"
            with open(temp_page_path, "wb") as temp_pdf_file:
                pdf_writer.write(temp_pdf_file)

            logger.info(f"Processing page {page_num + 1}/{num_pages} from {temp_page_path}...")
            try:
                if method == "ollama":
                    document_text = extract_text_with_doctr(temp_page_path)
                    if document_text:
                        extracted_fields, validation_result = extract_fields_with_ollama(document_text)
                        if extracted_fields:
                            all_extracted_data.extend(extracted_fields)
                            all_raw_responses.append(json.dumps(extracted_fields))
                            if validation_result:
                                validation_results.append(validation_result)
                        else:
                            all_raw_responses.append(f"No response from Ollama for page {page_num + 1}")
                    else:
                        all_raw_responses.append(f"DocTR failed to extract text from page {page_num + 1}")
                else:  # vertex_ai
                    response = process_document(temp_page_path, get_extraction_prompt())
                    extracted_page_data = extract_fields(response)
                    all_extracted_data.extend(extracted_page_data)
                    if response:
                        all_raw_responses.append(response.get("raw_response", ""))
                    else:
                        all_raw_responses.append(f"No response from process_document for page {page_num + 1}")

            except Exception as e:
                logger.error(f"Error processing page {page_num + 1}: {e}")
                all_raw_responses.append(f"Error processing page {page_num + 1}: {str(e)}")
            finally:
                if os.path.exists(temp_page_path):
                    os.remove(temp_page_path) # Clean up temporary file

    except Exception as e:
        logger.error(f"Error reading PDF or splitting pages for {file_path}: {e}")
        # Fallback: try processing the whole file directly if splitting fails
        logger.info(f"Attempting to process file {file_path} as a whole due to splitting error...")
        try:
            if method == "ollama":
                document_text = extract_text_with_doctr(file_path)
                if document_text:
                    extracted_fields, validation_result = extract_fields_with_ollama(document_text)
                    if extracted_fields:
                        all_extracted_data.extend(extracted_fields)
                        all_raw_responses.append(json.dumps(extracted_fields))
                        if validation_result:
                            validation_results.append(validation_result)
                    else:
                        all_raw_responses.append("No response from Ollama for whole file (fallback).")
                else:
                    all_raw_responses.append("DocTR failed to extract text from whole file (fallback).")
            else:  # vertex_ai
                response = process_document(file_path, get_extraction_prompt())
                extracted_page_data = extract_fields(response)
                all_extracted_data.extend(extracted_page_data)
                if response:
                    all_raw_responses.append(response.get("raw_response", ""))
                else:
                                        all_raw_responses.append("No response from process_document for whole file (fallback).")
        except Exception as fallback_e:
            logger.error(f"Error processing whole file during fallback: {fallback_e}")
            all_raw_responses.append(f"Error processing whole file during fallback: {str(fallback_e)}")

    result = {
        "case_id": case_id,
        "document_type": "kyc",
        "file_path": file_path,
        "extracted_data": all_extracted_data, # Aggregated list from all pages
        "method_used": method,
        "raw_response": "\n--- Page Separator ---\n".join(all_raw_responses) # Join raw responses
    }
    
    if method == "ollama" and validation_results:
        result["validation_results"] = validation_results
        
    return result

def extract_details(case_id, file_path):
    """
    Extract details from a KYC document with fallback mechanism
    
    Args:
        case_id: Unique identifier for the document case
        file_path: Path to the document file
        
    Returns:
        Dictionary with extracted fields
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # First try with Ollama + DocTR
    logger.info(f"Attempting extraction with Ollama + DocTR for {file_path}")
    
    # Extract text using docTR OCR for the whole document first
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
        elif validation_result and validation_result.get("confidence_score", 0) < 70:
            logger.info(f"Low confidence score from Ollama: {validation_result.get('confidence_score')}, falling back to Vertex AI")
            use_vertex_fallback = True
        elif validation_result and validation_result.get("digit_confidence", 0) < 80:
            logger.info(f"Low digit confidence score from Ollama: {validation_result.get('digit_confidence')}, falling back to Vertex AI")
            use_vertex_fallback = True
    
    # Process based on chosen method
    if use_vertex_fallback:
        logger.info("Using Vertex AI for extraction")
        result = extract_details_from_all_pages(case_id, file_path, method="vertex_ai")
        print(f"✅ Extraction completed using Vertex AI")
    else:
        logger.info("Using Ollama for extraction")
        # For KYC documents, we still want to process page by page to catch multiple entries
        # that might be spread across different pages
        result = extract_details_from_all_pages(case_id, file_path, method="ollama")
        print(f"✅ Extraction completed using Ollama")
    
    # Post-processing for digit verification
    if result and "extracted_data" in result and result["extracted_data"]:
        for entry in result["extracted_data"]:
            # Double-check Aadhaar number format
            aadhaar = entry.get("aadhaarNumber", "")
            if aadhaar:
                # Ensure it follows the standard format
                aadhaar_match = re.search(r'xxxx xxxx (\d{4})', aadhaar)
                if not aadhaar_match:
                    # Try to fix the format
                    digits = re.findall(r'\d', aadhaar)
                    if len(digits) >= 4:
                        last_four = ''.join(digits[-4:])
                        entry["aadhaarNumber"] = f"xxxx xxxx {last_four}"
                    elif len(digits) > 0:
                        # If less than 4 digits, pad with x
                        last_digits = ''.join(digits)
                        padding = 'x' * (4 - len(last_digits))
                        entry["aadhaarNumber"] = f"xxxx xxxx {padding}{last_digits}"
    
    return result

def main():
    """Test function for KYC document extraction"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract details from a KYC document')
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

