from google import genai
from google.genai import types
import os
import json
import base64
import fitz  # PyMuPDF for PDF handling
import tempfile
from PIL import Image
import io

def process_document(file_path, project_id="cifcl-poc-ai", location="asia-south1", credentials_path="database/cifcl-poc-ai.json"):
    """
    Process a cheque document (image or PDF) using Gemini to extract structured information.
    
    Args:
        file_path: Path to the cheque image or PDF file
        project_id: Google Cloud project ID
        location: Google Cloud region
        credentials_path: Path to the service account credentials file
        
    Returns:
        Structured JSON data with extracted cheque information
    """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    client = genai.Client(vertexai=True, project=project_id, location=location)
    model = "gemini-1.5-pro-002"
    
    # Determine file type
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Process based on file type
    if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
        # For image files, process directly
        return process_image_file(file_path, client, model)
    elif file_extension == '.pdf':
        # For PDF files, process each page
        return process_pdf_file(file_path, client, model)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def process_image_file(image_path, client, model):
    """Process an image file using Gemini."""
    # Read and encode the image
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Create the prompt for cheque information extraction
    prompt = get_cheque_extraction_prompt()
    
    # Process the image
    return generate_content_with_image(client, model, prompt, image_base64, "image/jpeg")

def process_pdf_file(pdf_path, client, model):
    """Process a PDF file using Gemini."""
    try:
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        
        # Check if PDF has pages
        if len(pdf_document) == 0:
            raise ValueError("The PDF document contains no pages")
        
        # For multi-page PDFs, we'll process the first page
        # You could modify this to process all pages or specific pages
        page = pdf_document.load_page(0)
        
        # Convert PDF page to image using a temporary file
        temp_img_path = os.path.join(tempfile.gettempdir(), f"pdf_page_{os.path.basename(pdf_path)}.png")
        
        # Render page to pixmap
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution for better OCR
        
        # Save pixmap to temporary file
        pix.save(temp_img_path)
        
        # Read the temporary file and encode as base64
        with open(temp_img_path, "rb") as img_file:
            image_bytes = img_file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Clean up temporary file
        try:
            os.remove(temp_img_path)
        except:
            pass
        
        # Create the prompt for cheque information extraction
        prompt = get_cheque_extraction_prompt()
        
        # Close the PDF
        pdf_document.close()
        
        # Process the image
        return generate_content_with_image(client, model, prompt, image_base64, "image/png")
    
    except Exception as e:
        raise RuntimeError(f"Error processing PDF: {e}")

def get_cheque_extraction_prompt():
    """Return the prompt for cheque information extraction."""
    return """
    Extract all visible text present in the bank cheque image with high accuracy. Focus on capturing key details and output them in a structured JSON object format, with the following field names and precise formatting:
    
    payeeName: Full name of the payee.
    date: The cheque issuance date in 'ddmmyyyy' format, identified accurately without variations.
    chequeNumber: Unique cheque number.
    accountNumber: Complete bank account number.
    bankName: Full name of the bank.
    branch: Branch name and location, capturing all address details.
    amountInWords: Cheque amount as written in words (e.g., "Ten Thousand Only").
    amountInNumbers: Cheque amount as represented in numeric form (e.g., "10000").
    signatureName: Name on the signature line.
    micrCode: MICR code exactly as displayed on the cheque.
    ifscCode: Bank's IFSC code.
    
    Ensure the JSON object format is clean, with each extracted field labeled precisely by the above field names, capturing all required information accurately. Do not omit or miss any details. If a field is not found, return it as an empty string or null to maintain consistency. Structure and separate each extracted field clearly to avoid any ambiguity in the output format.
    
    Look carefully at all parts of the cheque including the top, bottom, and corners. Pay special attention to small text that might contain account numbers, MICR codes, or IFSC codes. For dates, convert any format to ddmmyyyy.
    
    Return ONLY the JSON object without any additional text, explanations, or markdown formatting.
    """

def generate_content_with_image(client, model, prompt, image_base64, mime_type):
    """Generate content using Gemini with an image."""
    try:
        # Configure the model
        config = types.GenerateContentConfig(
            temperature=0.1,  # Lower temperature for more deterministic output
            top_p=0.95,
            max_output_tokens=8192,
            response_mime_type="application/json",
            safety_settings=[types.SafetySetting(category=c, threshold="OFF") for c in [
                "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_DANGEROUS_CONTENT", 
                "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_HARASSMENT"]]
        )
        
        # Create the content parts
        content_parts = [
            {"text": prompt},
            {"inline_data": {"mime_type": mime_type, "data": image_base64}}
        ]
        
        # Generate content
        response = client.models.generate_content(
            model=model,
            contents=content_parts,
            config=config
        )
        
        print("Processing complete!")
        
        # Extract and parse the JSON response
        try:
            # Clean the response text to ensure it's valid JSON
            response_text = response.text.strip()
            
            # If response is wrapped in code blocks, extract just the JSON
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0].strip()
                
            # Parse the JSON
            result = json.loads(response_text)
            
            # Return both the structured data and the raw response
            return {
                "structured_data": result,
                "raw_response": response.text
            }
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print("Raw response:", response.text)
            return {
                "structured_data": None,
                "raw_response": response.text,
                "error": f"Failed to parse JSON: {e}"
            }
        
    except Exception as e:
        print(f"Error generating response: {e}")
        raise RuntimeError(f"Error generating response: {e}")

def process_multi_page_pdf(pdf_path, client, model, max_pages=3):
    """Process a multi-page PDF, extracting information from each page."""
    # Initialize Gemini client if not provided
    if client is None or model is None:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "database/cifcl-poc-ai.json"
        client = genai.Client(vertexai=True, project="cifcl-poc-ai", location="asia-south1")
        model = "gemini-1.5-pro-002"
    
    # Open the PDF
    pdf_document = fitz.open(pdf_path)
    
    # Check if PDF has pages
    if len(pdf_document) == 0:
        raise ValueError("The PDF document contains no pages")
    
    # Process each page up to max_pages
    results = []
    for page_num in range(min(len(pdf_document), max_pages)):
        print(f"Processing page {page_num + 1} of {len(pdf_document)}")
        
        # Load the page
        page = pdf_document.load_page(page_num)
        
        # Create a temporary file for the page image
        temp_img_path = os.path.join(tempfile.gettempdir(), f"pdf_page_{page_num}_{os.path.basename(pdf_path)}.png")
        
        try:
            # Render page to pixmap
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            
            # Save pixmap to temporary file
            pix.save(temp_img_path)
            
            # Read the temporary file and encode as base64
            with open(temp_img_path, "rb") as img_file:
                image_bytes = img_file.read()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Create the prompt for cheque information extraction
            prompt = get_cheque_extraction_prompt()
            
            # Process the image
            result = generate_content_with_image(client, model, prompt, image_base64, "image/png")
            result["page_number"] = page_num + 1
            results.append(result)
            
        except Exception as e:
            print(f"Error processing page {page_num + 1}: {e}")
            results.append({
                "structured_data": None,
                "raw_response": f"Error: {e}",
                "page_number": page_num + 1,
                "error": str(e)
            })
        
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
            except:
                pass
    
    # Close the PDF
    pdf_document.close()
    
    return results

if __name__ == "__main__":
    # Get the file path from user input
    file_path = input("Enter the path to the cheque document (image or PDF): ")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
    else:
        try:
            # Check if it's a multi-page PDF
            if file_path.lower().endswith('.pdf'):
                pdf_document = fitz.open(file_path)
                num_pages = len(pdf_document)
                pdf_document.close()
                
                if num_pages > 1:
                    process_all = input(f"The PDF has {num_pages} pages. Process all pages? (y/n, default: n): ").lower()
                    
                    if process_all == 'y':
                        # Process all pages
                        max_pages = int(input(f"Enter maximum number of pages to process (default: 3): ") or 3)
                        results = process_multi_page_pdf(file_path, None, None, max_pages)
                        
                        # Save the results to a file
                        output_file = f"{os.path.splitext(file_path)[0]}_extracted.json"
                        with open(output_file, 'w') as f:
                            json.dump(results, f, indent=2)
                        print(f"\nResults saved to {output_file}")
                        
                        # Display results for each page
                        for result in results:
                            if result.get("structured_data"):
                                print(f"\nPage {result['page_number']} - Extracted Information:")
                                print("-" * 50)
                                print(json.dumps(result["structured_data"], indent=2))
                            else:
                                print(f"\nPage {result['page_number']} - Failed to extract structured data")
                                if "error" in result:
                                    print(f"Error: {result['error']}")
                        
                        exit()
            
            # Process single file (image or single-page PDF)
            result = process_document(file_path)
            
            if result.get("structured_data"):
                print("\nExtracted Cheque Information:")
                print("-" * 50)
                print(json.dumps(result["structured_data"], indent=2))
                
                # Save the results to a file
                output_file = f"{os.path.splitext(file_path)[0]}_extracted.json"
                with open(output_file, 'w') as f:
                    json.dump(result["structured_data"], f, indent=2)
                print(f"\nResults saved to {output_file}")
            else:
                print("\nFailed to extract structured data. Raw response:")
                print("-" * 50)
                print(result.get("raw_response", "No response"))
                if "error" in result:
                    print(f"Error: {result['error']}")
        except Exception as e:
            print(f"Error processing document: {e}")
