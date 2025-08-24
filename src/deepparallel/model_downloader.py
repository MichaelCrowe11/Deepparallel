"""
Model Download and Cloud Storage Manager for Deep Parallel Genesis
"""

import asyncio
import logging
import os
import boto3
from google.cloud import storage as gcs
from azure.storage.blob import BlobServiceClient
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
from huggingface_hub import snapshot_download, login
import torch
from transformers import AutoModel, AutoTokenizer
import time

from .config import GenesisConfig

logger = logging.getLogger(__name__)


class ModelDownloadManager:
    """
    Manages downloading and storing large language models
    from various sources to cloud storage
    """
    
    def __init__(self, config: GenesisConfig):
        self.config = config
        self.local_cache = config.model.cache_dir
        self.cloud_providers = self._initialize_cloud_providers()
        self.model_registry = self._load_model_registry()
        
        # Ensure cache directory exists
        Path(self.local_cache).mkdir(parents=True, exist_ok=True)
    
    def _initialize_cloud_providers(self) -> Dict[str, Any]:
        """Initialize cloud storage clients"""
        providers = {}
        
        # AWS S3
        if self.config.storage.aws_access_key_id:
            providers['aws'] = boto3.client(
                's3',
                aws_access_key_id=self.config.storage.aws_access_key_id,
                aws_secret_access_key=self.config.storage.aws_secret_access_key,
                region_name=self.config.storage.aws_region
            )
        
        # Google Cloud Storage
        if self.config.storage.gcp_service_account_key:
            providers['gcp'] = gcs.Client.from_service_account_json(
                self.config.storage.gcp_service_account_key
            )
        
        # Azure Blob Storage
        if self.config.storage.azure_connection_string:
            providers['azure'] = BlobServiceClient.from_connection_string(
                self.config.storage.azure_connection_string
            )
        
        return providers
    
    def _load_model_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load comprehensive model registry for open source models"""
        return {
            # Meta Llama Models
            "llama-3.1-8b": {
                "hf_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "size_gb": 16.0,
                "parameters": "8B",
                "domain": "general",
                "quality_score": 0.92,
                "specialties": ["reasoning", "conversation", "code"],
                "context_length": 128000,
                "license": "llama3.1"
            },
            "llama-3.1-70b": {
                "hf_name": "meta-llama/Meta-Llama-3.1-70B-Instruct",
                "size_gb": 140.0,
                "parameters": "70B",
                "domain": "general",
                "quality_score": 0.95,
                "specialties": ["advanced_reasoning", "complex_tasks", "research"],
                "context_length": 128000,
                "license": "llama3.1"
            },
            "llama-3.1-405b": {
                "hf_name": "meta-llama/Meta-Llama-3.1-405B-Instruct",
                "size_gb": 810.0,
                "parameters": "405B",
                "domain": "general",
                "quality_score": 0.98,
                "specialties": ["frontier_reasoning", "scientific_research", "expert_tasks"],
                "context_length": 128000,
                "license": "llama3.1"
            },
            
            # Google Gemma Models
            "gemma-2-9b": {
                "hf_name": "google/gemma-2-9b-it",
                "size_gb": 18.0,
                "parameters": "9B",
                "domain": "general",
                "quality_score": 0.89,
                "specialties": ["efficiency", "safety", "multilingual"],
                "context_length": 8192,
                "license": "gemma"
            },
            "gemma-2-27b": {
                "hf_name": "google/gemma-2-27b-it",
                "size_gb": 54.0,
                "parameters": "27B",
                "domain": "general",
                "quality_score": 0.93,
                "specialties": ["reasoning", "math", "science"],
                "context_length": 8192,
                "license": "gemma"
            },
            
            # Alibaba Qwen Models
            "qwen2.5-7b": {
                "hf_name": "Qwen/Qwen2.5-7B-Instruct",
                "size_gb": 14.0,
                "parameters": "7B",
                "domain": "general",
                "quality_score": 0.88,
                "specialties": ["multilingual", "code", "math"],
                "context_length": 32768,
                "license": "qwen"
            },
            "qwen2.5-14b": {
                "hf_name": "Qwen/Qwen2.5-14B-Instruct",
                "size_gb": 28.0,
                "parameters": "14B",
                "domain": "general",
                "quality_score": 0.91,
                "specialties": ["reasoning", "multilingual", "technical"],
                "context_length": 32768,
                "license": "qwen"
            },
            "qwen2.5-72b": {
                "hf_name": "Qwen/Qwen2.5-72B-Instruct",
                "size_gb": 144.0,
                "parameters": "72B",
                "domain": "general",
                "quality_score": 0.94,
                "specialties": ["advanced_reasoning", "research", "expert_tasks"],
                "context_length": 32768,
                "license": "qwen"
            },
            
            # Microsoft Phi Models
            "phi-3.5-mini": {
                "hf_name": "microsoft/Phi-3.5-mini-instruct",
                "size_gb": 7.5,
                "parameters": "3.8B",
                "domain": "general",
                "quality_score": 0.84,
                "specialties": ["efficiency", "edge_deployment", "mobile"],
                "context_length": 128000,
                "license": "mit"
            },
            
            # Mistral Models
            "mistral-7b": {
                "hf_name": "mistralai/Mistral-7B-Instruct-v0.3",
                "size_gb": 14.0,
                "parameters": "7B",
                "domain": "general",
                "quality_score": 0.87,
                "specialties": ["efficiency", "reasoning", "conversation"],
                "context_length": 32768,
                "license": "apache2"
            },
            "mixtral-8x7b": {
                "hf_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "size_gb": 90.0,
                "parameters": "8x7B",
                "domain": "general",
                "quality_score": 0.93,
                "specialties": ["mixture_of_experts", "efficiency", "multilingual"],
                "context_length": 32768,
                "license": "apache2"
            },
            
            # Specialized Science Models
            "codegemma-7b": {
                "hf_name": "google/codegemma-7b-it",
                "size_gb": 14.0,
                "parameters": "7B",
                "domain": "code",
                "quality_score": 0.88,
                "specialties": ["code_generation", "programming", "debugging"],
                "context_length": 8192,
                "license": "gemma"
            },
            "deepseek-math-7b": {
                "hf_name": "deepseek-ai/deepseek-math-7b-instruct",
                "size_gb": 14.0,
                "parameters": "7B",
                "domain": "mathematics",
                "quality_score": 0.90,
                "specialties": ["mathematics", "problem_solving", "proofs"],
                "context_length": 4096,
                "license": "deepseek"
            },
            "biogpt": {
                "hf_name": "microsoft/BioGPT-Large",
                "size_gb": 6.0,
                "parameters": "1.5B",
                "domain": "biology",
                "quality_score": 0.85,
                "specialties": ["biomedical", "literature", "research"],
                "context_length": 1024,
                "license": "mit"
            },
            
            # Research and Academic Models
            "galactica-6.7b": {
                "hf_name": "facebook/galactica-6.7b",
                "size_gb": 13.0,
                "parameters": "6.7B",
                "domain": "science",
                "quality_score": 0.86,
                "specialties": ["scientific_literature", "citations", "research"],
                "context_length": 2048,
                "license": "cc-by-nc-4.0"
            },
            "falcon-7b": {
                "hf_name": "tiiuae/falcon-7b-instruct",
                "size_gb": 14.0,
                "parameters": "7B",
                "domain": "general",
                "quality_score": 0.85,
                "specialties": ["multilingual", "efficiency", "reasoning"],
                "context_length": 2048,
                "license": "apache2"
            }
        }
    
    async def download_all_genesis_models(
        self, 
        target_ensemble_size: int = 12,
        size_constraint_gb: Optional[float] = None,
        exclude_licenses: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Download a comprehensive ensemble of models for Genesis Engine
        """
        logger.info(f"Downloading Genesis model ensemble (target size: {target_ensemble_size})")
        
        # Filter models based on constraints
        available_models = self._filter_models(size_constraint_gb, exclude_licenses)
        
        # Select optimal ensemble
        selected_models = self._select_optimal_ensemble(
            available_models, target_ensemble_size
        )
        
        download_results = {}
        total_size = 0.0
        
        # Download models in parallel (with concurrency limit)
        semaphore = asyncio.Semaphore(3)  # Limit concurrent downloads
        
        download_tasks = [
            self._download_single_model(model_id, model_info, semaphore)
            for model_id, model_info in selected_models.items()
        ]
        
        results = await asyncio.gather(*download_tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            model_id = list(selected_models.keys())[i]
            if isinstance(result, Exception):
                logger.error(f"Failed to download {model_id}: {result}")
                download_results[model_id] = {
                    "status": "failed",
                    "error": str(result),
                    "local_path": None,
                    "cloud_urls": {}
                }
            else:
                download_results[model_id] = result
                total_size += selected_models[model_id]["size_gb"]
        
        logger.info(f"Downloaded {len(download_results)} models, total size: {total_size:.1f} GB")
        
        # Save ensemble manifest
        await self._save_ensemble_manifest(download_results)
        
        return download_results
    
    def _filter_models(
        self, 
        size_constraint_gb: Optional[float], 
        exclude_licenses: Optional[List[str]]
    ) -> Dict[str, Dict[str, Any]]:
        """Filter models based on constraints"""
        
        filtered = {}
        exclude_licenses = exclude_licenses or []
        
        for model_id, model_info in self.model_registry.items():
            # Size constraint
            if size_constraint_gb and model_info["size_gb"] > size_constraint_gb:
                continue
            
            # License constraint
            if model_info["license"] in exclude_licenses:
                continue
            
            filtered[model_id] = model_info
        
        return filtered
    
    def _select_optimal_ensemble(
        self, 
        available_models: Dict[str, Dict[str, Any]], 
        target_size: int
    ) -> Dict[str, Dict[str, Any]]:
        """Select optimal ensemble of models for maximum diversity and capability"""
        
        # Sort by quality score
        sorted_models = sorted(
            available_models.items(),
            key=lambda x: x[1]["quality_score"],
            reverse=True
        )
        
        selected = {}
        domains_covered = set()
        specialties_covered = set()
        total_params = 0
        
        # Selection strategy: maximize diversity while maintaining quality
        for model_id, model_info in sorted_models:
            if len(selected) >= target_size:
                break
            
            # Check if this model adds value
            adds_domain = model_info["domain"] not in domains_covered
            adds_specialty = not set(model_info["specialties"]).issubset(specialties_covered)
            high_quality = model_info["quality_score"] > 0.85
            
            # Include flagship models regardless
            is_flagship = any(keyword in model_id for keyword in [
                "llama-3.1-405b", "llama-3.1-70b", "qwen2.5-72b", "gemma-2-27b"
            ])
            
            if adds_domain or adds_specialty or high_quality or is_flagship:
                selected[model_id] = model_info
                domains_covered.add(model_info["domain"])
                specialties_covered.update(model_info["specialties"])
                
                # Parse parameter count
                param_str = model_info["parameters"]
                if "B" in param_str:
                    total_params += float(param_str.replace("B", ""))
                elif "x" in param_str:  # Mixture of experts
                    parts = param_str.split("x")
                    total_params += float(parts[0]) * float(parts[1].replace("B", ""))
        
        logger.info(f"Selected {len(selected)} models covering {len(domains_covered)} domains")
        logger.info(f"Total parameters: {total_params:.1f}B")
        logger.info(f"Specialties covered: {sorted(specialties_covered)}")
        
        return selected
    
    async def _download_single_model(
        self, 
        model_id: str, 
        model_info: Dict[str, Any], 
        semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """Download a single model with progress tracking"""
        
        async with semaphore:
            logger.info(f"Starting download: {model_id} ({model_info['size_gb']:.1f} GB)")
            start_time = time.time()
            
            hf_name = model_info["hf_name"]
            local_path = Path(self.local_cache) / model_id
            
            try:
                # Check if already downloaded
                if local_path.exists() and self._verify_model_integrity(local_path):
                    logger.info(f"Model {model_id} already exists and verified")
                    return await self._process_existing_model(model_id, model_info, local_path)
                
                # Download from Hugging Face
                logger.info(f"Downloading {hf_name} to {local_path}")
                
                # Use snapshot_download for full model
                downloaded_path = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: snapshot_download(
                        repo_id=hf_name,
                        cache_dir=str(local_path),
                        local_files_only=False,
                        revision="main"
                    )
                )
                
                # Verify download
                if not self._verify_model_integrity(downloaded_path):
                    raise Exception("Model integrity check failed")
                
                # Upload to cloud storage
                cloud_urls = await self._upload_to_cloud_storage(
                    model_id, downloaded_path, model_info
                )
                
                download_time = time.time() - start_time
                download_speed = model_info["size_gb"] / (download_time / 3600)  # GB/hour
                
                logger.info(f"Successfully downloaded {model_id} in {download_time:.1f}s "
                          f"({download_speed:.1f} GB/hour)")
                
                return {
                    "status": "success",
                    "model_id": model_id,
                    "local_path": str(downloaded_path),
                    "cloud_urls": cloud_urls,
                    "download_time": download_time,
                    "size_gb": model_info["size_gb"],
                    "verified": True,
                    "hf_name": hf_name
                }
                
            except Exception as e:
                logger.error(f"Failed to download {model_id}: {e}")
                raise
    
    def _verify_model_integrity(self, model_path: Path) -> bool:
        """Verify model integrity"""
        
        model_path = Path(model_path)
        
        # Check if path exists
        if not model_path.exists():
            return False
        
        # Check for essential files
        essential_files = ["config.json"]
        pytorch_files = list(model_path.glob("*.bin")) + list(model_path.glob("*.safetensors"))
        
        # Must have config and at least one model file
        has_config = any((model_path / f).exists() for f in essential_files)
        has_weights = len(pytorch_files) > 0
        
        return has_config and has_weights
    
    async def _process_existing_model(
        self, 
        model_id: str, 
        model_info: Dict[str, Any], 
        local_path: Path
    ) -> Dict[str, Any]:
        """Process already downloaded model"""
        
        # Check cloud storage status
        cloud_urls = await self._check_cloud_storage(model_id)
        
        # Upload to cloud if not already there
        if not cloud_urls:
            cloud_urls = await self._upload_to_cloud_storage(
                model_id, local_path, model_info
            )
        
        return {
            "status": "exists",
            "model_id": model_id,
            "local_path": str(local_path),
            "cloud_urls": cloud_urls,
            "size_gb": model_info["size_gb"],
            "verified": True,
            "hf_name": model_info["hf_name"]
        }
    
    async def _upload_to_cloud_storage(
        self, 
        model_id: str, 
        local_path: Path, 
        model_info: Dict[str, Any]
    ) -> Dict[str, str]:
        """Upload model to cloud storage providers"""
        
        cloud_urls = {}
        
        # Create archive for upload (optional compression)
        archive_path = await self._create_model_archive(local_path, model_id)
        
        # Upload to each available provider
        upload_tasks = []
        
        if 'aws' in self.cloud_providers:
            upload_tasks.append(
                self._upload_to_aws(model_id, archive_path)
            )
        
        if 'gcp' in self.cloud_providers:
            upload_tasks.append(
                self._upload_to_gcp(model_id, archive_path)
            )
        
        if 'azure' in self.cloud_providers:
            upload_tasks.append(
                self._upload_to_azure(model_id, archive_path)
            )
        
        # Execute uploads in parallel
        if upload_tasks:
            upload_results = await asyncio.gather(*upload_tasks, return_exceptions=True)
            
            providers = ['aws', 'gcp', 'azure'][:len(upload_tasks)]
            for i, result in enumerate(upload_results):
                provider = providers[i]
                if isinstance(result, Exception):
                    logger.error(f"Failed to upload {model_id} to {provider}: {result}")
                else:
                    cloud_urls[provider] = result
        
        # Clean up archive
        if archive_path and archive_path.exists():
            archive_path.unlink()
        
        return cloud_urls
    
    async def _create_model_archive(self, model_path: Path, model_id: str) -> Optional[Path]:
        """Create compressed archive of model (optional)"""
        
        # For large models, we might want to upload directory structure directly
        # For now, return the original path
        return model_path
    
    async def _upload_to_aws(self, model_id: str, model_path: Path) -> str:
        """Upload model to AWS S3"""
        
        s3_client = self.cloud_providers['aws']
        bucket_name = self.config.storage.aws_bucket_name
        
        if model_path.is_dir():
            # Upload directory structure
            base_key = f"genesis-models/{model_id}/"
            
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(model_path)
                    s3_key = base_key + str(relative_path)
                    
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: s3_client.upload_file(
                            str(file_path), bucket_name, s3_key
                        )
                    )
            
            return f"s3://{bucket_name}/{base_key}"
        else:
            # Upload single file
            s3_key = f"genesis-models/{model_id}/{model_path.name}"
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: s3_client.upload_file(
                    str(model_path), bucket_name, s3_key
                )
            )
            
            return f"s3://{bucket_name}/{s3_key}"
    
    async def _upload_to_gcp(self, model_id: str, model_path: Path) -> str:
        """Upload model to Google Cloud Storage"""
        
        gcs_client = self.cloud_providers['gcp']
        bucket_name = self.config.storage.gcp_bucket_name
        bucket = gcs_client.bucket(bucket_name)
        
        if model_path.is_dir():
            # Upload directory structure
            base_key = f"genesis-models/{model_id}/"
            
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(model_path)
                    blob_name = base_key + str(relative_path)
                    
                    blob = bucket.blob(blob_name)
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: blob.upload_from_filename(str(file_path))
                    )
            
            return f"gs://{bucket_name}/{base_key}"
        else:
            # Upload single file
            blob_name = f"genesis-models/{model_id}/{model_path.name}"
            blob = bucket.blob(blob_name)
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: blob.upload_from_filename(str(model_path))
            )
            
            return f"gs://{bucket_name}/{blob_name}"
    
    async def _upload_to_azure(self, model_id: str, model_path: Path) -> str:
        """Upload model to Azure Blob Storage"""
        
        blob_service = self.cloud_providers['azure']
        container_name = self.config.storage.azure_container_name
        
        if model_path.is_dir():
            # Upload directory structure
            base_key = f"genesis-models/{model_id}/"
            
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(model_path)
                    blob_name = base_key + str(relative_path)
                    
                    blob_client = blob_service.get_blob_client(
                        container=container_name, blob=blob_name
                    )
                    
                    with open(file_path, 'rb') as data:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: blob_client.upload_blob(data, overwrite=True)
                        )
            
            return f"https://{blob_service.account_name}.blob.core.windows.net/{container_name}/{base_key}"
        else:
            # Upload single file
            blob_name = f"genesis-models/{model_id}/{model_path.name}"
            blob_client = blob_service.get_blob_client(
                container=container_name, blob=blob_name
            )
            
            with open(model_path, 'rb') as data:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: blob_client.upload_blob(data, overwrite=True)
                )
            
            return f"https://{blob_service.account_name}.blob.core.windows.net/{container_name}/{blob_name}"
    
    async def _check_cloud_storage(self, model_id: str) -> Dict[str, str]:
        """Check if model exists in cloud storage"""
        
        cloud_urls = {}
        
        # Check AWS S3
        if 'aws' in self.cloud_providers:
            aws_url = await self._check_aws_storage(model_id)
            if aws_url:
                cloud_urls['aws'] = aws_url
        
        # Check GCP
        if 'gcp' in self.cloud_providers:
            gcp_url = await self._check_gcp_storage(model_id)
            if gcp_url:
                cloud_urls['gcp'] = gcp_url
        
        # Check Azure
        if 'azure' in self.cloud_providers:
            azure_url = await self._check_azure_storage(model_id)
            if azure_url:
                cloud_urls['azure'] = azure_url
        
        return cloud_urls
    
    async def _check_aws_storage(self, model_id: str) -> Optional[str]:
        """Check if model exists in AWS S3"""
        try:
            s3_client = self.cloud_providers['aws']
            bucket_name = self.config.storage.aws_bucket_name
            prefix = f"genesis-models/{model_id}/"
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: s3_client.list_objects_v2(
                    Bucket=bucket_name, Prefix=prefix, MaxKeys=1
                )
            )
            
            if response.get('Contents'):
                return f"s3://{bucket_name}/{prefix}"
        except Exception as e:
            logger.debug(f"Error checking AWS storage for {model_id}: {e}")
        
        return None
    
    async def _check_gcp_storage(self, model_id: str) -> Optional[str]:
        """Check if model exists in Google Cloud Storage"""
        try:
            gcs_client = self.cloud_providers['gcp']
            bucket_name = self.config.storage.gcp_bucket_name
            bucket = gcs_client.bucket(bucket_name)
            prefix = f"genesis-models/{model_id}/"
            
            blobs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: list(bucket.list_blobs(prefix=prefix, max_results=1))
            )
            
            if blobs:
                return f"gs://{bucket_name}/{prefix}"
        except Exception as e:
            logger.debug(f"Error checking GCP storage for {model_id}: {e}")
        
        return None
    
    async def _check_azure_storage(self, model_id: str) -> Optional[str]:
        """Check if model exists in Azure Blob Storage"""
        try:
            blob_service = self.cloud_providers['azure']
            container_name = self.config.storage.azure_container_name
            prefix = f"genesis-models/{model_id}/"
            
            blobs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: list(blob_service.list_blobs(
                    container_name, name_starts_with=prefix, max_results=1
                ))
            )
            
            if blobs:
                return f"https://{blob_service.account_name}.blob.core.windows.net/{container_name}/{prefix}"
        except Exception as e:
            logger.debug(f"Error checking Azure storage for {model_id}: {e}")
        
        return None
    
    async def _save_ensemble_manifest(self, download_results: Dict[str, Dict[str, Any]]):
        """Save ensemble manifest for tracking"""
        
        manifest = {
            "genesis_ensemble_version": "1.0",
            "created_at": time.time(),
            "total_models": len(download_results),
            "total_size_gb": sum(
                result.get("size_gb", 0) 
                for result in download_results.values()
            ),
            "models": download_results
        }
        
        manifest_path = Path(self.local_cache) / "genesis_ensemble_manifest.json"
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Saved ensemble manifest to {manifest_path}")
    
    async def get_model_status(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of downloaded models"""
        
        status = {
            "local_cache": str(self.local_cache),
            "total_models": 0,
            "total_size_gb": 0.0,
            "models": {}
        }
        
        # Load manifest if exists
        manifest_path = Path(self.local_cache) / "genesis_ensemble_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                status.update(manifest)
        
        # If specific model requested
        if model_id:
            if model_id in status.get("models", {}):
                return {"model_id": model_id, **status["models"][model_id]}
            else:
                return {"model_id": model_id, "status": "not_found"}
        
        return status
    
    async def cleanup_local_cache(self, keep_recent: int = 5) -> Dict[str, Any]:
        """Cleanup local cache, keeping only recent models"""
        
        cache_path = Path(self.local_cache)
        if not cache_path.exists():
            return {"status": "no_cache", "cleaned": 0, "freed_gb": 0.0}
        
        # Get model directories sorted by modification time
        model_dirs = [
            d for d in cache_path.iterdir() 
            if d.is_dir() and d.name != ".git"
        ]
        model_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Keep recent models, remove others
        to_remove = model_dirs[keep_recent:]
        freed_gb = 0.0
        cleaned_count = 0
        
        for model_dir in to_remove:
            # Calculate size before removal
            size_gb = sum(
                f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
            ) / (1024**3)
            
            # Remove directory
            shutil.rmtree(model_dir, ignore_errors=True)
            freed_gb += size_gb
            cleaned_count += 1
        
        logger.info(f"Cleaned {cleaned_count} models, freed {freed_gb:.1f} GB")
        
        return {
            "status": "success",
            "cleaned": cleaned_count,
            "freed_gb": round(freed_gb, 2),
            "remaining": len(model_dirs) - cleaned_count
        }
