import os
import shutil
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass


class DataIngestionConfig:
    data_dir: str = os.path.join('artifacts', 'raw_data')
    source_data_healthy: str = os.path.join('research', 'data', 'Healthy')
    source_data_coccidiosis: str = os.path.join('research', 'data', 'Coccidiosis')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Data Ingestion method starts")
            os.makedirs(self.ingestion_config.data_dir, exist_ok=True)


            healthy_dir = os.path.join(self.ingestion_config.data_dir,'Healthy')
            coccidiosis_dir = os.path.join(self.ingestion_config.data_dir,'Coccidiosis')

            os.makedirs(healthy_dir, exist_ok=True)
            os.makedirs(coccidiosis_dir, exist_ok=True)


            source_files_healthy = os.listdir(self.ingestion_config.source_data_healthy)
            for file in source_files_healthy:
                source_file_path = os.path.join(self.ingestion_config.source_data_healthy, file)
                destination_file_path = os.path.join(healthy_dir, file)
                shutil.copy(source_file_path, destination_file_path)
            logging.info("Healthy data ingestion completed")


            source_files_coccidiosis = os.listdir(self.ingestion_config.source_data_coccidiosis)
            for file in source_files_coccidiosis:
                source_file_path = os.path.join(self.ingestion_config.source_data_coccidiosis, file)
                destination_file_path = os.path.join(coccidiosis_dir, file)
                shutil.copy(source_file_path, destination_file_path)
            logging.info("Coccidiosis data ingestion completed")

            return self.ingestion_config.data_dir

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()
