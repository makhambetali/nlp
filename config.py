import os
from dotenv import load_dotenv


load_dotenv()

class BaseConfig:
    PORT = 8000
    HOST = '0.0.0.0'

    DATABASE_URL = os.getenv("DATABASE_URL", "mongodb://127.0.0.1:27017")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "mydatabase")

    MODEL_NAME = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-large")

    INDEX_FILE = os.getenv("INDEX_FILE", "medical_annoy_index.ann")

class DevConfig(BaseConfig):
    DEBUG = True

class ProdConfig(BaseConfig):
    DEBUG = False

configs = {
    "prod": ProdConfig,
    "dev": DevConfig,
    "default": DevConfig
}

config = configs[os.getenv("ENV", "default")]
