"""
Model Management and Cloud Storage System for Deep Parallel Genesis
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    BitsAndBytesConfig
)
from huggingface_hub import hf_hub_download, snapshot_download
import boto3
from google.cloud import storage as gcs
from azure.storage.blob import BlobServiceClient

from .config import GenesisConfig, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a downloaded/cached model"""
    name: str
    path: str
    size_gb: float
    download_time: float
    last_used: str
    config: Dict[str, Any]


class ModelManager:
    """Manages downloading, caching, and loading of models from various sources"""
    
    def __init__(self, config: GenesisConfig):
        self.config = config
        self.model_config = config.models
        self.cache_dir = Path(self.model_config.cloud_storage["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cloud storage clients
        self.cloud_clients = self._initialize_cloud_clients()
        
        # Model registry for tracking downloaded models
        self.model_registry_path = self.cache_dir / "model_registry.json"
        self.model_registry = self._load_model_registry()
        
        logger.info(f"ModelManager initialized with cache dir: {self.cache_dir}")
    
    def _initialize_cloud_clients(self) -> Dict[str, Any]:
        """Initialize cloud storage clients"""
        clients = {}
        
        # AWS S3
        try:
            clients["aws"] = boto3.client('s3')
            logger.info("AWS S3 client initialized")
        except Exception as e:
            logger.warning(f"Could not initialize AWS client: {e}")
        
        # Google Cloud Storage
        try:
            clients["gcp"] = gcs.Client()
            logger.info("GCP Storage client initialized")
        except Exception as e:
            logger.warning(f"Could not initialize GCP client: {e}")
        
        # Azure Blob Storage
        try:
            connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if connection_string:
                clients["azure"] = BlobServiceClient.from_connection_string(connection_string)
                logger.info("Azure Blob Storage client initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Azure client: {e}")
        
        return clients
    
    def _load_model_registry(self) -> Dict[str, ModelInfo]:
        """Load the model registry from disk"""
        if self.model_registry_path.exists():
            with open(self.model_registry_path, 'r') as f:
                data = json.load(f)
                return {
                    name: ModelInfo(**info) for name, info in data.items()
                }
        return {}
    
    def _save_model_registry(self):
        """Save the model registry to disk"""
        data = {
            name: {
                "name": info.name,
                "path": info.path,
                "size_gb": info.size_gb,
                "download_time": info.download_time,
                "last_used": info.last_used,
                "config": info.config
            }
            for name, info in self.model_registry.items()
        }
        with open(self.model_registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def download_all_base_models(self, force_redownload: bool = False):
        """Download all configured base models"""
        logger.info("Starting download of all base models...")
        
        for model_name, model_id in self.model_config.base_models.items():
            try:
                self.download_model(model_name, model_id, force_redownload)
            except Exception as e:
                logger.error(f"Failed to download {model_name}: {e}")
        
        logger.info("Completed downloading base models")
    
    def download_model(
        self, 
        model_name: str, 
        model_id: str, 
        force_redownload: bool = False
    ) -> str:
        """Download a specific model"""
        
        # Check if already cached
        if not force_redownload and model_name in self.model_registry:
            model_path = self.model_registry[model_name].path
            if os.path.exists(model_path):
                logger.info(f"Model {model_name} already cached at {model_path}")
                return model_path
        
        logger.info(f"Downloading model {model_name} ({model_id})...")
        
        model_cache_dir = self.cache_dir / model_name
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        try:
            # Download using HuggingFace Hub
            snapshot_download(
                repo_id=model_id,
                cache_dir=str(model_cache_dir),
                use_auth_token=self.model_config.cloud_storage.get("use_auth_token", True),
                resume_download=True
            )
            
            download_time = time.time() - start_time
            size_gb = self._calculate_directory_size(model_cache_dir) / (1024**3)
            
            # Update registry
            self.model_registry[model_name] = ModelInfo(
                name=model_name,
                path=str(model_cache_dir),
                size_gb=size_gb,
                download_time=download_time,
                last_used=time.strftime("%Y-%m-%d %H:%M:%S"),
                config={"model_id": model_id, "source": "huggingface"}
            )
            
            self._save_model_registry()
            
            logger.info(f"Downloaded {model_name} ({size_gb:.2f} GB) in {download_time:.2f}s")
            return str(model_cache_dir)
            
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            raise
    
    def load_model_and_tokenizer(
        self, 
        model_name: str,
        load_config: Optional[Dict[str, Any]] = None
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load a model and tokenizer from cache"""
        
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not found in registry. Download it first.")
        
        model_path = self.model_registry[model_name].path
        config = load_config or self.model_config.load_config
        
        logger.info(f"Loading model {model_name} from {model_path}")
        
        # Configure quantization if requested
        quantization_config = None
        if config.get("load_in_4bit", False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, config.get("bnb_4bit_compute_dtype", "float16")),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif config.get("load_in_8bit", False):
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=config.get("trust_remote_code", True)
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, config.get("torch_dtype", "auto")) if config.get("torch_dtype") != "auto" else "auto",
            device_map=config.get("device_map", "auto"),
            trust_remote_code=config.get("trust_remote_code", True),
            quantization_config=quantization_config,
            use_flash_attention_2=config.get("use_flash_attention", True)
        )
        
        # Update last used time
        self.model_registry[model_name].last_used = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save_model_registry()
        
        logger.info(f"Successfully loaded {model_name}")
        return model, tokenizer
    
    def upload_to_cloud(
        self, 
        model_name: str, 
        cloud_provider: str = "aws",
        bucket_name: Optional[str] = None
    ):
        """Upload a cached model to cloud storage"""
        
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not found in registry")
        
        bucket_name = bucket_name or self.model_config.cloud_storage["bucket_name"]
        model_path = self.model_registry[model_name].path
        
        logger.info(f"Uploading {model_name} to {cloud_provider}://{bucket_name}")
        
        if cloud_provider == "aws" and "aws" in self.cloud_clients:
            self._upload_to_s3(model_path, bucket_name, model_name)
        elif cloud_provider == "gcp" and "gcp" in self.cloud_clients:
            self._upload_to_gcs(model_path, bucket_name, model_name)
        elif cloud_provider == "azure" and "azure" in self.cloud_clients:
            self._upload_to_azure(model_path, bucket_name, model_name)
        else:
            raise ValueError(f"Cloud provider {cloud_provider} not available or not configured")
    
    def _upload_to_s3(self, model_path: str, bucket_name: str, model_name: str):
        """Upload model to AWS S3"""
        s3_client = self.cloud_clients["aws"]
        
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                s3_key = f"models/{model_name}/{os.path.relpath(file_path, model_path)}"
                
                s3_client.upload_file(file_path, bucket_name, s3_key)
                logger.debug(f"Uploaded {file_path} to s3://{bucket_name}/{s3_key}")
    
    def _upload_to_gcs(self, model_path: str, bucket_name: str, model_name: str):
        """Upload model to Google Cloud Storage"""
        client = self.cloud_clients["gcp"]
        bucket = client.bucket(bucket_name)
        
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                blob_name = f"models/{model_name}/{os.path.relpath(file_path, model_path)}"
                
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(file_path)
                logger.debug(f"Uploaded {file_path} to gs://{bucket_name}/{blob_name}")
    
    def _upload_to_azure(self, model_path: str, container_name: str, model_name: str):
        """Upload model to Azure Blob Storage"""
        blob_service_client = self.cloud_clients["azure"]
        
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                blob_name = f"models/{model_name}/{os.path.relpath(file_path, model_path)}"
                
                with open(file_path, "rb") as data:
                    blob_service_client.upload_blob(
                        container=container_name,
                        name=blob_name,
                        data=data,
                        overwrite=True
                    )
                logger.debug(f"Uploaded {file_path} to azure://{container_name}/{blob_name}")
    
    def _calculate_directory_size(self, path: Path) -> int:
        """Calculate total size of directory in bytes"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
    
    def list_models(self) -> Dict[str, ModelInfo]:
        """List all cached models"""
        return self.model_registry.copy()
    
    def delete_model(self, model_name: str):
        """Delete a cached model"""
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not found in registry")
        
        model_path = self.model_registry[model_name].path
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            logger.info(f"Deleted model {model_name} from {model_path}")
        
        del self.model_registry[model_name]
        self._save_model_registry()
    
    def cleanup_cache(self, keep_most_recent: int = 5):
        """Clean up old cached models, keeping only the most recently used"""
        if len(self.model_registry) <= keep_most_recent:
            return
        
        # Sort by last used time
        sorted_models = sorted(
            self.model_registry.items(),
            key=lambda x: x[1].last_used,
            reverse=True
        )
        
        # Delete older models
        for model_name, _ in sorted_models[keep_most_recent:]:
            try:
                self.delete_model(model_name)
                logger.info(f"Cleaned up old model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to clean up {model_name}: {e}")


class EnsembleModelManager:
    """Manages multiple models for ensemble reasoning"""
    
    def __init__(self, config: GenesisConfig):
        self.config = config
        self.model_manager = ModelManager(config)
        self.loaded_models: Dict[str, tuple] = {}  # model_name -> (model, tokenizer)
    
    def load_ensemble_models(self, model_names: List[str]):
        """Load multiple models for ensemble use"""
        logger.info(f"Loading ensemble models: {model_names}")
        
        for model_name in model_names:
            try:
                model, tokenizer = self.model_manager.load_model_and_tokenizer(model_name)
                self.loaded_models[model_name] = (model, tokenizer)
                logger.info(f"Loaded {model_name} for ensemble")
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
    
    def get_ensemble_response(
        self, 
        prompt: str,
        model_names: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Get responses from multiple models"""
        model_names = model_names or list(self.loaded_models.keys())
        responses = {}
        
        for model_name in model_names:
            if model_name in self.loaded_models:
                try:
                    model, tokenizer = self.loaded_models[model_name]
                    
                    # Tokenize input
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                    
                    # Generate response
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            temperature=0.1,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    # Decode response
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = response[len(prompt):].strip()  # Remove input prompt
                    
                    responses[model_name] = response
                    
                except Exception as e:
                    logger.error(f"Error getting response from {model_name}: {e}")
                    responses[model_name] = f"Error: {str(e)}"
        
        return responses
