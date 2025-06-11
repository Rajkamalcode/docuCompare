import json
import csv
import os
import re
from datetime import datetime
import difflib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment variables for offline mode
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Try to import sentence-transformers with fallback options
model = None
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    
    # Try to load the model from a local path
    try:
        model_path = r'C:\Users\intern-rajkamal\.cache\torch\hub\sentence-transformers\all-MiniLM-L6-v2'
        model = SentenceTransformer(model_path, local_files_only=True)
        logger.info(f"Successfully loaded model from {model_path}")
        
        # Test the model to make sure it works
        test_embedding = model.encode("This is a test sentence")
        logger.info(f"Model test successful. Embedding shape: {test_embedding.shape}")
    except Exception as e:
        logger.error(f"Error loading model from specific path: {e}")
        
        # Try alternative model loading approaches
        try:
            # Try loading a simpler model that might be available locally
            model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
            logger.info("Successfully loaded model using default path")
        except Exception as e2:
            logger.error(f"Error loading alternative model: {e2}")
            model = None
except ImportError as e:
    logger.error(f"Could not import sentence-transformers: {e}")
    logger.info("Will use simple text comparison instead of semantic matching")
    
    # Define a simple util module for fallback
    class SimpleUtil:
        @staticmethod
        def pytorch_cos_sim(vec1, vec2):
            # Simple dot product similarity
            return sum(a*b for a, b in zip(vec1, vec2)) / (
                (sum(a*a for a in vec1) ** 0.5) * 
                (sum(b*b for b in vec2) ** 0.5)
            )
    
    util = SimpleUtil()
    
import config

# Global variable to store RAPID_SYSTEM data
RAPID_SYSTEM = {}

def set_rapid_system_data(data):
    """
    Set RAPID_SYSTEM data for comparison
    
    Args:
        data: Dictionary containing RAPID_SYSTEM data
    """
    global RAPID_SYSTEM
    RAPID_SYSTEM = data
    logger.info(f"RAPID_SYSTEM data set: {len(RAPID_SYSTEM)} document types")

def load_comparison_rules():
    """
    Load comparison rules from the CSV file
    
    Returns:
        Dictionary of comparison rules by document type
    """
    rules = {}
    
    # Path to the comparison rules CSV
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'comparsion.csv')
    
    if not os.path.exists(csv_path):
        print(f"Warning: Comparison rules file not found at {csv_path}")
        return rules
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        current_doc_type = None
        
        for row in reader:
            if len(row) < 3:
                continue
                
            # Check if this is a new document type
            if row[0]:
                # Handle special case for "Memorandum of title deposits"
                if "Memorandum of title" in row[1]:
                    current_doc_type = "memorandum_of_title"
                else:
                    current_doc_type = row[1].lower().replace(' ', '_')
                
                if current_doc_type not in rules:
                    rules[current_doc_type] = {}
            
            # Skip header rows or empty rows
            if not current_doc_type or not row[2]:
                continue
            
            # Add the rule
            field_name = row[2].lower().replace(' ', '_')
            comparison_rule = row[3] if len(row) > 3 else ""
            
            rules[current_doc_type][field_name] = {
                'rule': comparison_rule
            }
    
    return rules

def compare_documents(case_id, documents_by_type):
    """
    Compare documents for a case based on predefined rules
    
    Args:
        case_id: Unique identifier for the case
        documents_by_type: Dictionary of documents organized by type
        
    Returns:
        Dictionary of comparison results
    """
    # Load comparison rules
    rules = load_comparison_rules()
    
    # Initialize results
    results = {}
    
    # Process each document type with rules
    for doc_type, rules_for_type in rules.items():
        # Skip if we don't have this document type
        if doc_type not in documents_by_type:
            continue
            
        doc = documents_by_type[doc_type]
        extracted_data = doc.get('extracted_data', {})
        
        # Initialize results for this document type
        results[doc_type] = {}
        
        # Process each field with rules
        for field_name, rule_data in rules_for_type.items():
            rule = rule_data.get('rule', '')
            
            # Get the field value
            field_value = get_nested_field_value(extracted_data, field_name)
            
            # Initialize result for this field
            results[doc_type][field_name] = {
                'value': field_value,
                'rule': rule
            }
            
            # Process the rule
            if rule.startswith('Compare with '):
                # Extract the target document type
                target_doc_type = rule.replace('Compare with ', '').lower().replace(' ', '_')
                
                # Check if comparing with RAPID_SYSTEM
                if target_doc_type == 'rapid_system':
                    # Check if we have RAPID_SYSTEM data for this document type
                    if doc_type in RAPID_SYSTEM and 'fields' in RAPID_SYSTEM[doc_type]:
                        rapid_data = RAPID_SYSTEM[doc_type]['fields']
                        
                        # Try to find a matching field in RAPID_SYSTEM data
                        target_value = find_matching_field(rapid_data, field_name)
                        
                        # Compare the values
                        comparison_result = compare_values(field_value, target_value)
                        
                        # Update the result
                        results[doc_type][field_name].update({
                            'status': 'compared',
                            'target_document': 'RAPID_SYSTEM',
                            'target_value': target_value,
                            'result': comparison_result
                        })
                    else:
                        # RAPID_SYSTEM data not found
                        results[doc_type][field_name].update({
                            'status': 'error',
                            'message': f"RAPID_SYSTEM data not found for '{doc_type}'"
                        })
                # Check if we have the target document
                elif target_doc_type in documents_by_type:
                    target_doc = documents_by_type[target_doc_type]
                    target_data = target_doc.get('extracted_data', {})
                    
                    # Try to find a matching field in the target document
                    target_value = find_matching_field(target_data, field_name)
                    
                    # Compare the values
                    comparison_result = compare_values(field_value, target_value)
                    
                    # Update the result
                    results[doc_type][field_name].update({
                        'status': 'compared',
                        'target_document': target_doc_type,
                        'target_value': target_value,
                        'result': comparison_result
                    })
                else:
                    # Target document not found
                    results[doc_type][field_name].update({
                        'status': 'error',
                        'message': f"Target document '{target_doc_type}' not found"
                    })
            elif rule.startswith('Should be '):
                # Extract the expected value
                expected_value = rule.replace('Should be ', '')
                
                # Compare with the expected value
                comparison_result = compare_values(field_value, expected_value)
                
                # Update the result
                results[doc_type][field_name].update({
                    'status': 'compared',
                    'target_value': expected_value,
                    'result': comparison_result
                })
            elif rule.startswith('Availability of '):
                # Check for availability
                is_available = bool(field_value)
                
                # Update the result
                results[doc_type][field_name].update({
                    'status': 'compared',
                    'target_value': 'Available',
                    'result': {
                        'overall_match': is_available,
                        'best_confidence': 1.0 if is_available else 0.0
                    }
                })
            elif rule.startswith('The date should be '):
                # Date comparison logic
                date_rule = rule.replace('The date should be ', '')
                
                # Parse the date from the field value
                field_date = parse_date(field_value)
                
                if field_date:
                    # Compare based on the rule
                    if 'after' in date_rule or 'greater than' in date_rule:
                        target_doc_type = date_rule.split(' ')[-1].lower().replace(' ', '_')
                        
                        if target_doc_type in documents_by_type:
                            target_doc = documents_by_type[target_doc_type]
                            target_data = target_doc.get('extracted_data', {})
                            
                            # Find a date field in the target document
                            target_date_value = find_date_field(target_data)
                            target_date = parse_date(target_date_value)
                            
                            if target_date:
                                # Check if field_date is after target_date
                                is_after = field_date > target_date
                                
                                # Update the result
                                results[doc_type][field_name].update({
                                    'status': 'compared',
                                    'target_value': target_date_value,
                                    'result': {
                                        'overall_match': is_after,
                                        'best_confidence': 1.0 if is_after else 0.0
                                    }
                                })
                            else:
                                results[doc_type][field_name].update({
                                    'status': 'error',
                                    'message': f"Could not parse date from target document '{target_doc_type}'"
                                })
                        else:
                            results[doc_type][field_name].update({
                                'status': 'error',
                                'message': f"Target document '{target_doc_type}' not found"
                            })
                else:
                    results[doc_type][field_name].update({
                        'status': 'error',
                        'message': "Could not parse date from field value"
                    })
            else:
                # No specific comparison rule
                results[doc_type][field_name].update({
                    'status': 'info',
                    'message': "No specific comparison performed"
                })
    
    return results

def get_nested_field_value(data, field_name):
    """
    Get a value from a nested dictionary using a field name
    
    Args:
        data: Dictionary to search
        field_name: Field name (can be nested with dots)
        
    Returns:
        Field value or None if not found
    """
    if not data:
        return None
        
    # Handle nested fields (e.g., "dpn.borrowersSignatures")
    if '.' in field_name:
        parts = field_name.split('.')
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
                
        return current
    
    # Try direct match
    if field_name in data:
        return data[field_name]
    
    # Try case-insensitive match
    for key in data:
        if key.lower() == field_name.lower():
            return data[key]
    
    # Try to match by removing underscores and spaces
    normalized_field = field_name.lower().replace('_', '').replace(' ', '')
    for key in data:
        if key.lower().replace('_', '').replace(' ', '') == normalized_field:
            return data[key]
    
    return None

def find_matching_field(data, field_name):
    """
    Find a matching field in the data
    
    Args:
        data: Dictionary to search
        field_name: Field name to match
        
    Returns:
        Field value or None if not found
    """
    # First try direct match
    direct_match = get_nested_field_value(data, field_name)
    if direct_match is not None:
        return direct_match
    
    # Try to find a semantically similar field
    normalized_field = field_name.lower().replace('_', ' ')
    
    best_match = None
    best_score = 0
    
    for key in data:
        normalized_key = key.lower().replace('_', ' ')
        
        # Calculate similarity score
        score = difflib.SequenceMatcher(None, normalized_field, normalized_key).ratio()
        
        if score > best_score and score > 0.7:  # Threshold for considering a match
            best_score = score
            best_match = key
    
    if best_match:
        return data[best_match]
    
    return None

def find_date_field(data):
    """
    Find a field that contains a date
    
    Args:
        data: Dictionary to search
        
    Returns:
        Date value or None if not found
    """
    date_keywords = ['date', 'created', 'updated', 'issued']
    
    # First look for keys that contain date keywords
    for key in data:
        if any(keyword in key.lower() for keyword in date_keywords):
            return data[key]
    
    # Then look for values that might be dates
    for value in data.values():
        if isinstance(value, str) and parse_date(value):
            return value
    
    return None

def parse_date(date_str):
    """
    Parse a date string into a datetime object
    
    Args:
        date_str: Date string
        
    Returns:
        datetime object or None if parsing fails
    """
    if not date_str or not isinstance(date_str, str):
        return None
    
    # Try common date formats
    formats = [
        '%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%Y/%m/%d',
        '%d/%m/%y', '%d-%m-%y', '%y-%m-%d', '%y/%m/%d',
        '%d %b %Y', '%d %B %Y', '%b %d, %Y', '%B %d, %Y',
        '%d.%m.%Y', '%Y.%m.%d'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    # Try to extract a date using regex
    date_patterns = [
        r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})',  # dd/mm/yyyy or mm/dd/yyyy
        r'(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})'     # yyyy/mm/dd
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, date_str)
        if match:
            groups = match.groups()
            
            # Try to determine the format based on the values
            if len(groups) == 3:
                if len(groups[0]) == 4:  # yyyy/mm/dd
                    try:
                        year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                        return datetime(year, month, day)
                    except ValueError:
                        continue
                else:  # dd/mm/yyyy or mm/dd/yyyy
                    try:
                        # Assume dd/mm/yyyy
                        day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                        if year < 100:
                            year += 2000 if year < 50 else 1900
                        return datetime(year, month, day)
                    except ValueError:
                        try:
                            # Try mm/dd/yyyy
                            month, day, year = int(groups[0]), int(groups[1]), int(groups[2])
                            if year < 100:
                                year += 2000 if year < 50 else 1900
                            return datetime(year, month, day)
                        except ValueError:
                            continue
    
    return None

def compare_values(value1, value2):
    """
    Compare two values using both exact and semantic matching
    
    Args:
        value1: First value
        value2: Second value
        
    Returns:
        Dictionary with comparison results
    """
    if value1 is None or value2 is None:
        return {
            'exact_match': False,
            'semantic_match': False,
            'best_confidence': 0.0,
            'overall_match': False
        }
    
    # Convert to strings for comparison
    str_value1 = str(value1).strip().lower()
    str_value2 = str(value2).strip().lower()
    
    # Exact match
    exact_match = str_value1 == str_value2
    
    # If we have an exact match, no need for semantic matching
    if exact_match:
        return {
            'exact_match': True,
            'semantic_match': True,
            'similarity_score': 1.0,
            'best_confidence': 1.0,
            'overall_match': True
        }
    
    # Try semantic match if model is available
    if model is not None:
        try:
            # Encode the strings
            embeddings = model.encode([str_value1, str_value2])
            
            # Calculate cosine similarity
            similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
            
            # Check if similarity exceeds threshold
            semantic_match = similarity >= getattr(config, 'SEMANTIC_MATCH_THRESHOLD', 0.8)
            
            return {
                'exact_match': exact_match,
                'semantic_match': semantic_match,
                'similarity_score': similarity,
                'best_confidence': similarity,
                'overall_match': exact_match or semantic_match
            }
        except Exception as e:
            logger.error(f"Error in semantic matching: {e}")
            # Fall through to fallback method
    
    # Fallback to simpler comparison if semantic matching fails or is unavailable
    try:
        # Use sequence matcher for string similarity
        similarity = difflib.SequenceMatcher(None, str_value1, str_value2).ratio()
        semantic_match = similarity >= 0.8
        
        return {
            'exact_match': exact_match,
            'semantic_match': semantic_match,
            'similarity_score': similarity,
            'best_confidence': similarity,
            'overall_match': exact_match or semantic_match,
            'method': 'fallback'
        }
    except Exception as e:
        logger.error(f"Error in fallback comparison: {e}")
        return {
            'exact_match': exact_match,
            'semantic_match': False,
            'similarity_score': 0.0,
            'best_confidence': 1.0 if exact_match else 0.0,
            'overall_match': exact_match,
            'error': str(e)
        }
