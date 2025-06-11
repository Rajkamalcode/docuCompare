from datetime import datetime

class Document:
    """Base class for document data models"""
    
    def __init__(self, case_id, file_path=None):
        self.case_id = case_id
        self.file_path = file_path
        self.extracted_data = {}
        self.raw_response = None
        self.extraction_date = datetime.utcnow()
        
    def to_dict(self):
        """Convert document data to dictionary"""
        return {
            "case_id": self.case_id,
            "file_path": self.file_path,
            "extracted_data": self.extracted_data,
            "extraction_date": self.extraction_date
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create document object from dictionary"""
        doc = cls(data["case_id"], data.get("file_path"))
        doc.extracted_data = data.get("extracted_data", {})
        doc.extraction_date = data.get("extraction_date", datetime.utcnow())
        return doc

class SanctionLetter(Document):
    """Sanction Letter document model"""
    
    def __init__(self, case_id, file_path=None):
        super().__init__(case_id, file_path)
        self.document_type = "sanction_letter"

class LegalReport(Document):
    """Legal Report document model"""
    
    def __init__(self, case_id, file_path=None):
        super().__init__(case_id, file_path)
        self.document_type = "legal_report"

class RepaymentKit(Document):
    """Repayment Kit document model"""
    
    def __init__(self, case_id, file_path=None):
        super().__init__(case_id, file_path)
        self.document_type = "repayment_kit"

class KYC(Document):
    """KYC document model"""
    
    def __init__(self, case_id, file_path=None):
        super().__init__(case_id, file_path)
        self.document_type = "kyc"

class VettingReport(Document):
    """Vetting Report document model"""
    
    def __init__(self, case_id, file_path=None):
        super().__init__(case_id, file_path)
        self.document_type = "vetting_report"

class Annexure(Document):
    """Annexure document model"""
    
    def __init__(self, case_id, file_path=None):
        super().__init__(case_id, file_path)
        self.document_type = "annexure"

class MemorandumOfTitle(Document):
    """Memorandum of Title document model"""
    
    def __init__(self, case_id, file_path=None):
        super().__init__(case_id, file_path)
        self.document_type = "memorandum_of_title"

class Agreement(Document):
    """Agreement document model"""
    
    def __init__(self, case_id, file_path=None):
        super().__init__(case_id, file_path)
        self.document_type = "agreement"