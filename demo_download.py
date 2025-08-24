#!/usr/bin/env python3
"""
Quick demo script to test Deep Parallel Genesis model downloading
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from deepparallel.config import GenesisConfig
from deepparallel.model_downloader import ModelDownloadManager

async def demo_model_download():
    """Demo the model download functionality"""
    print("🧬 Deep Parallel Genesis - Model Download Demo")
    print("=" * 50)
    
    # Create config
    config = GenesisConfig()
    
    # Override cache directory for demo
    config.model.cache_dir = "/tmp/deepparallel_demo_cache"
    
    # Initialize downloader
    downloader = ModelDownloadManager(config)
    
    print(f"📁 Cache directory: {config.model.cache_dir}")
    print(f"🎯 Available models: {len(downloader.model_registry)}")
    
    # Show available models
    print("\n📋 Available Models:")
    print("-" * 50)
    for model_id, info in list(downloader.model_registry.items())[:10]:  # Show first 10
        print(f"• {model_id:20} | {info['parameters']:>8} | {info['size_gb']:>6.1f}GB | {info['domain']}")
    
    print(f"... and {len(downloader.model_registry) - 10} more models")
    
    # Test model selection algorithm
    print("\n🎯 Testing Model Selection Algorithm:")
    print("-" * 50)
    
    # Filter small models for demo
    small_models = downloader._filter_models(size_constraint_gb=20.0, exclude_licenses=[])
    print(f"Models under 20GB: {len(small_models)}")
    
    # Select optimal small ensemble
    selected = downloader._select_optimal_ensemble(small_models, target_size=3)
    
    print(f"\n🔍 Selected 3-model ensemble:")
    total_size = 0
    for model_id, info in selected.items():
        print(f"• {model_id:20} | {info['quality_score']:.2f} | {info['size_gb']:>6.1f}GB | {info['specialties']}")
        total_size += info['size_gb']
    
    print(f"\nTotal ensemble size: {total_size:.1f}GB")
    
    # Demo cloud storage check (won't actually connect without credentials)
    print(f"\n☁️  Cloud Storage Providers:")
    for provider in ['aws', 'gcp', 'azure']:
        configured = provider in downloader.cloud_providers
        status = "✅ Configured" if configured else "❌ Not configured"
        print(f"• {provider.upper():5}: {status}")
    
    print(f"\n💡 To actually download models:")
    print(f"1. Configure cloud storage credentials in .env")
    print(f"2. Run: deepparallel download-models --count 3 --size-limit 20")
    print(f"3. Or run: python demo_download.py --actually-download")
    
    # Check if user wants to actually download
    if "--actually-download" in sys.argv:
        print(f"\n🚀 Starting actual download of 1 small model for demo...")
        
        # Download just one small model for demo
        small_model = {"phi-3.5-mini": downloader.model_registry["phi-3.5-mini"]}
        
        try:
            results = await downloader.download_all_genesis_models(
                target_ensemble_size=1,
                size_constraint_gb=10.0,
                exclude_licenses=[]
            )
            
            print(f"\n✅ Download results:")
            for model_id, result in results.items():
                print(f"• {model_id}: {result.get('status', 'unknown')}")
                if result.get('local_path'):
                    print(f"  📁 Local: {result['local_path']}")
                if result.get('cloud_urls'):
                    print(f"  ☁️  Cloud: {list(result['cloud_urls'].keys())}")
                    
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            print(f"This is expected without proper HuggingFace access or cloud credentials")
    
    print(f"\n🎉 Demo complete!")

if __name__ == "__main__":
    asyncio.run(demo_model_download())
