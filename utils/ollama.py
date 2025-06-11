import requests
import json
import logging

logger = logging.getLogger(__name__)

# Default Ollama API URL - update this to the correct endpoint
DEFAULT_OLLAMA_URL = "http://10.9.52.21:11435/api/generate"  # Note: Changed from 11435 to 11434

def call_ollama_api(prompt, ollama_url=None, model_name="gemma3:12b-it-qat", step_name="Analysis"):
    """Call the Ollama API to generate text
    
    Args:
        prompt: The prompt to send to the model
        ollama_url: The URL of the Ollama API (optional)
        model_name: The name of the model to use
        step_name: A name for this step (for logging)
    
    Returns:
        The generated text
    """
    payload = {
        "model": model_name, 
        "prompt": prompt, 
        "stream": False, 
        "options": {
            "num_ctx": 32768, 
            "temperature": 0.15, 
            "repeat_penalty": 1.15, 
            "top_k": 30, 
            "top_p": 0.85
        }
    }
    
    full_response_text = ""
    api_url_to_call = ollama_url if ollama_url else DEFAULT_OLLAMA_URL
    
    logger.info(f"Calling Ollama ({step_name}) API ({api_url_to_call}) model '{model_name}'...")
    
    try:
        # First, check if the Ollama server is running
        base_url = api_url_to_call.rsplit('/', 2)[0]  # Get base URL without /api/generate
        try:
            health_check = requests.get(base_url, timeout=5)
            if health_check.status_code != 200:
                logger.error(f"Ollama server not responding correctly at {base_url}. Status: {health_check.status_code}")
                return None
        except Exception as e:
            logger.error(f"Ollama server not available at {base_url}: {e}")
            return None
            
        # Now try the actual API call
        response = requests.post(api_url_to_call, json=payload, timeout=700)
        
        # Log the response status and URL for debugging
        logger.info(f"Ollama API response status: {response.status_code} for URL: {api_url_to_call}")
        
        if response.status_code == 404:
            # Try alternative endpoints if the main one fails
            alternative_endpoints = [
                api_url_to_call.replace("/api/generate", "/api/chat"),
                api_url_to_call.replace("/api/generate", "/v1/chat/completions"),
                api_url_to_call.replace("/api/generate", "/v1/completions")
            ]
            
            for alt_endpoint in alternative_endpoints:
                logger.info(f"Trying alternative endpoint: {alt_endpoint}")
                try:
                    alt_response = requests.post(alt_endpoint, json=payload, timeout=700)
                    if alt_response.status_code == 200:
                        logger.info(f"Alternative endpoint successful: {alt_endpoint}")
                        response = alt_response
                        break
                except Exception as alt_e:
                    logger.warning(f"Alternative endpoint failed: {alt_endpoint} - {alt_e}")
            
            if response.status_code == 404:
                logger.error(f"All Ollama API endpoints returned 404. Check server configuration.")
                return None
        
        response.raise_for_status()
        data = response.json()
        
        logger.info(f"Ollama ({step_name}) API call successful.")
        
        if 'response' in data and isinstance(data['response'], str):
            full_response_text = data['response']
        elif 'message' in data and isinstance(data.get('message'), dict) and 'content' in data['message']:
            full_response_text = data['message']['content']
        elif 'content' in data and isinstance(data['content'], str) and 'choices' not in data:
            full_response_text = data['content']
        elif 'choices' in data and len(data['choices']) > 0:
            # Handle OpenAI-like response format
            choice = data['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                full_response_text = choice['message']['content']
            elif 'text' in choice:
                full_response_text = choice['text']
        elif isinstance(data, dict) and 'error' in data:
            logger.error(f"Ollama API error ({step_name}): {data['error']}")
            return None
        else:
            logger.warning(f"Unexpected Ollama response ({step_name}). Raw: {str(data)[:300]}...")
            possible_responses = [v for k, v in data.items() if isinstance(v, str) and len(v) > 20]
            if possible_responses:
                full_response_text = possible_responses[0]
                logger.info("Used fallback response extraction.")
            else:
                logger.error("Couldn't extract content.")
                return None
        
        return clean_processing_artifacts(full_response_text)
    
    except requests.exceptions.Timeout:
        logger.error(f"Ollama request ({step_name}) timed out.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama request error ({step_name}): {e}. Check server.")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Ollama response JSON decode error ({step_name}): {e}. Raw: {response.text[:500]}...")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in Ollama call ({step_name}): {e}.")
        return None

def clean_processing_artifacts(text):
    """Clean up any processing artifacts from the text"""
    if not text:
        return text
    
    # Remove any markdown code block markers
    text = text.replace("```json", "").replace("```", "")
    
    # Remove any leading/trailing whitespace
    text = text.strip()
    
    return text
