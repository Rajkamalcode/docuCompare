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
        document = {
            "case_id": case_id,
            "document_type": document_type,
            "extracted_data": extracted_data,
            "file_path": file_path,
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow()
        }
        
        # Check if document already exists
        existing_doc = self.collection.find_one({
            "case_id": case_id,
            "document_type": document_type
        })
        
        if existing_doc:
            # Update existing document
            result = self.collection.update_one(
                {"_id": existing_doc["_id"]},
                {
                    "$set": {
                        "extracted_data": extracted_data,
                        "file_path": file_path,
                        "updated_at": datetime.datetime.utcnow()
                    }
                }
            )
            return existing_doc["_id"]
        else:
            # Insert new document
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
        query = {"case_id": case_id}
        if document_type:
            query["document_type"] = document_type
            return self.collection.find_one(query)
        else:
            return list(self.collection.find(query))
    
    def get_all_cases(self):
        """
        Get a list of all unique case IDs
        
        Returns:
            List of case IDs
        """
        pipeline = [
            {"$group": {
                "_id": "$case_id",
                "document_count": {"$sum": 1},
                "last_updated": {"$max": "$updated_at"}
            }},
            {"$project": {
                "case_id": "$_id",
                "document_count": 1,
                "last_updated": 1,
                "_id": 0
            }}
        ]
        
        return list(self.collection.aggregate(pipeline))
    
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
