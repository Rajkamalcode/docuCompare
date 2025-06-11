from extractors import sanction_letter, legal_report, repayment_kit, kyc, vetting_report, annexure, memorandum_of_title, agreement

# Map document types to their respective extractors
EXTRACTORS = {
    "sanction_letter": sanction_letter,
    "legal_report": legal_report,
    "repayment_kit": repayment_kit,
    "kyc": kyc,
    "vetting_report": vetting_report,
    "annexure": annexure,
    "memorandum_of_title": memorandum_of_title,
    "agreement": agreement
}

def get_extractor(document_type):
    """
    Get the appropriate extractor module for a document type
    
    Args:
        document_type: Type of document to extract
        
    Returns:
        Extractor module
    """
    document_type = document_type.lower().replace(' ', '_')
    
    if document_type in EXTRACTORS:
        return EXTRACTORS[document_type]
    else:
        raise ValueError(f"No extractor available for document type: {document_type}")

def extract_document(case_id, document_type, file_path):
    """
    Extract details from a document using the appropriate extractor
    
    Args:
        case_id: Unique identifier for the document case
        document_type: Type of document to extract
        file_path: Path to the document file
        
    Returns:
        Dictionary with extracted fields
    """
    extractor = get_extractor(document_type)
    return extractor.extract_details(case_id, file_path)