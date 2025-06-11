from google import genai
from google.genai import types
import os
import json
import base64
import fitz  # PyMuPDF for PDF handling
import tempfile
from PIL import Image
import io
import docx
import config

def process_document(file_path, prompt, project_id=config.VERTEX_AI_PROJECT_ID, 
                    location=config.VERTEX_AI_LOCATION, 
                    credentials_path=config.CREDENTIALS_PATH):
    """
    Process a document (image, PDF, or DOCX) using Gemini to extract structured information.
    
    Args:
        file_path: Path to the document file
        prompt: The prompt to send to Gemini
        project_id: Google Cloud project ID
        location: Google Cloud region
        credentials_path: Path to the service account credentials file
        
    Returns:
        Structured JSON data with extracted information
    """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    client = genai.Client(vertexai=True, project=project_id, location=location)
    model = config.VERTEX_AI_MODEL
    
    # Determine file type
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Process based on file type
    if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
        # For image files, process directly
        return process_image_file(file_path, client, model, prompt)
    elif file_extension == '.pdf':
        # For PDF files, process the first page
        return process_pdf_file(file_path, client, model, prompt)
    elif file_extension in ['.doc', '.docx']:
        # For Word documents, convert to images and process
        return process_docx_file(file_path, client, model, prompt)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def process_image_file(image_path, client, model, prompt):
    """Process an image file using Gemini."""
    # Read and encode the image
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Process the image
    return generate_content_with_image(client, model, prompt, image_base64, "image/jpeg")

def process_pdf_file(pdf_path, client, model, prompt):
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
        
        # Close the PDF
        pdf_document.close()
        
        # Process the image
        return generate_content_with_image(client, model, prompt, image_base64, "image/png")
    
    except Exception as e:
        raise RuntimeError(f"Error processing PDF: {e}")

def process_docx_file(docx_path, client, model, prompt):
    """Process a DOCX file using Gemini."""
    try:
        # Create a temporary image file
        temp_img_path = os.path.join(tempfile.gettempdir(), f"docx_{os.path.basename(docx_path)}.png")
        
        # Extract text from DOCX
        doc = docx.Document(docx_path)
        text_content = "\n".join([para.text for para in doc.paragraphs])
        
        # Create an image with the text (simple approach)
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a blank image
        img = Image.new('RGB', (1000, 1500), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        
        # Use default font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw text on image
        d.text((20, 20), text_content, fill=(0, 0, 0), font=font)
        
        # Save the image
        img.save(temp_img_path)
        
        # Read and encode the image
        with open(temp_img_path, "rb") as img_file:
            image_bytes = img_file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Clean up temporary file
        try:
            os.remove(temp_img_path)
        except:
            pass
        
        # Process the image
        return generate_content_with_image(client, model, prompt, image_base64, "image/png")
    
    except Exception as e:
        raise RuntimeError(f"Error processing DOCX: {e}")

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