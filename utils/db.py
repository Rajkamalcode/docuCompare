import pymongo
from pymongo import MongoClient
import config
import datetime

class DocumentDB:
    def __init__(self):
        self.client = MongoClient(config.MONGO_URI)
        self.db = self.client[config.MONGO_DB]
        self.collection = self.db[config.MONGO_COLLECTION]
        self.comparison_collection = self.db['comparison_results']
        
    def store_document_data(self, case_id, document_type, extracted_data, file_path=None):
        """
        Store extracted document data in MongoDB
        
        Args:
            case_id: Unique identifier for the document case
            document_type: Type of document (e.g., 'sanction_letter', 'legal_report')
            extracted_data: Dictionary containing extracted fields
            file_path: Original file path (optional)
            
        Returns:
            MongoDB document ID
        """
        # Check if case already exists
        existing_case = self.collection.find_one({"case_id": case_id})
        
        if existing_case:
            # Update existing case with new document data
            update_data = {
                f"documents.{document_type}": {
                    "extracted_data": extracted_data,
                    "file_path": file_path,
                    "updated_at": datetime.datetime.utcnow()
                },
                "updated_at": datetime.datetime.utcnow()
            }
            
            result = self.collection.update_one(
                {"_id": existing_case["_id"]},
                {"$set": update_data}
            )
            return existing_case["_id"]
        else:
            # Create new case with document data
            document = {
                "case_id": case_id,
                "documents": {
                    document_type: {
                        "extracted_data": extracted_data,
                        "file_path": file_path,
                        "updated_at": datetime.datetime.utcnow()
                    }
                },
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            }
            
            result = self.collection.insert_one(document)
            return result.inserted_id
    
    def get_document_data(self, case_id, document_type=None):
        """
        Retrieve document data from MongoDB
        
        Args:
            case_id: Unique identifier for the document case
            document_type: Type of document (optional, if None returns all documents for the case)
            
        Returns:
            Document data or list of documents
        """
        case = self.collection.find_one({"case_id": case_id})
        
        if not case or "documents" not in case:
            return None
        
        if document_type:
            # Return specific document type
            if document_type in case["documents"]:
                doc_data = case["documents"][document_type]
                # Format to match the old structure for compatibility
                return {
                    "case_id": case_id,
                    "document_type": document_type,
                    "extracted_data": doc_data.get("extracted_data", {}),
                    "file_path": doc_data.get("file_path"),
                    "updated_at": doc_data.get("updated_at")
                }
            return None
        else:
            # Return all documents for the case
            documents = []
            for doc_type, doc_data in case["documents"].items():
                documents.append({
                    "case_id": case_id,
                    "document_type": doc_type,
                    "extracted_data": doc_data.get("extracted_data", {}),
                    "file_path": doc_data.get("file_path"),
                    "updated_at": doc_data.get("updated_at")
                })
            return documents
    
    def get_all_cases(self):
        """
        Get a list of all unique case IDs with metadata
        
        Returns:
            List of case IDs with document count and last updated time
        """
        cases = []
        for case in self.collection.find({}, {"case_id": 1, "documents": 1, "updated_at": 1}):
            doc_count = len(case.get("documents", {}))
            cases.append({
                "case_id": case["case_id"],
                "document_count": doc_count,
                "last_updated": case.get("updated_at")
            })
        return cases
    
    def store_comparison_results(self, case_id, comparison_data):
        """
        Store document comparison results in MongoDB
        
        Args:
            case_id: Unique identifier for the document case
            comparison_data: Dictionary containing comparison results
            
        Returns:
            MongoDB document ID
        """
        document = {
            "case_id": case_id,
            "comparison_data": comparison_data,
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow()
        }
        
        # Check if comparison already exists
        existing_doc = self.comparison_collection.find_one({
            "case_id": case_id
        })
        
        if existing_doc:
            # Update existing comparison
            result = self.comparison_collection.update_one(
                {"_id": existing_doc["_id"]},
                {
                    "$set": {
                        "comparison_data": comparison_data,
                        "updated_at": datetime.datetime.utcnow()
                    }
                }
            )
            return existing_doc["_id"]
        else:
            # Insert new comparison
            result = self.comparison_collection.insert_one(document)
            return result.inserted_id
    
    def get_comparison_results(self, case_id):
        """
        Retrieve comparison results from MongoDB
        
        Args:
            case_id: Unique identifier for the document case
            
        Returns:
            Comparison results document
        """
        return self.comparison_collection.find_one({"case_id": case_id})
