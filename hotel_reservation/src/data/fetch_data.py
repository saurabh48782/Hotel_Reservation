import os
import sys
from hotel_reservation.config import config
from google.cloud import storage
from hotel_reservation.src.logger import get_logger
from hotel_reservation.utils.custom_exception import CustomException

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.bucket_name = config['google_cloud_bucket']['bucket_name']
        self.file_name = config['google_cloud_bucket']['file_name']
        
        logger.info(f"Data ingestion will start from bucket {self.bucket_name}"
                    f" and file {self.file_name}")
    
    def download(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            
            blob.download_to_filename(sys.argv[1])

            logger.info(f"CSV data downloaded successfully at {sys.argv[1]}")
        
        except Exception as e:
            logger.error("Error occured while downloading CSV file from Google cloud bucket")
            raise CustomException("Failed to download CSV file", e)

    def run(self):
        try:
            logger.info("Data ingestion process started")
            self.download()
            logger.info("Data ingestion process completed successfully")
        
        except CustomException as ce:
            logger.error(f"Exception: {str(ce)}")
            
if __name__=='__main__':
    ingest = DataIngestion(config)
    ingest.run()
        
    