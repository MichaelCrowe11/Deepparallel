#!/bin/bash
# 
# Genesis Engine Setup Script
# Downloads and configures the optimal model ensemble for Deep Parallel Genesis
#

set -e

echo "🧬 Deep Parallel Genesis - Model Setup"
echo "======================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Install the package in development mode
echo "📦 Installing Deep Parallel in development mode..."
pip install -e .

# Check available storage space
echo "💾 Checking storage space..."
available_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
echo "Available space: ${available_space}GB"

if [ "$available_space" -lt 200 ]; then
    echo "⚠️  Warning: Less than 200GB available. Consider cleanup or use size limits."
    echo "   You can use --size-limit flag to limit individual model sizes."
fi

# Create cache directory
echo "📁 Creating model cache directory..."
mkdir -p ~/.cache/deepparallel/models

# Set up environment variables if .env doesn't exist
if [ ! -f ".env" ]; then
    echo "🔧 Creating environment configuration..."
    cp .env.example .env
    echo "📝 Please edit .env file with your API keys and cloud storage credentials"
fi

# Check for cloud storage configuration
echo "☁️  Checking cloud storage configuration..."
if [ -f ".env" ]; then
    source .env
    cloud_configured=false
    
    if [ ! -z "$AWS_ACCESS_KEY_ID" ]; then
        echo "✅ AWS S3 configured"
        cloud_configured=true
    fi
    
    if [ ! -z "$GCP_SERVICE_ACCOUNT_KEY" ]; then
        echo "✅ Google Cloud Storage configured"
        cloud_configured=true
    fi
    
    if [ ! -z "$AZURE_CONNECTION_STRING" ]; then
        echo "✅ Azure Blob Storage configured"
        cloud_configured=true
    fi
    
    if [ "$cloud_configured" = false ]; then
        echo "⚠️  No cloud storage configured - models will only be stored locally"
    fi
fi

echo ""
echo "🚀 Setup complete! Available commands:"
echo ""
echo "# Download a compact model ensemble (good for testing)"
echo "deepparallel download-models --count 3 --size-limit 20"
echo ""
echo "# Download the full Genesis ensemble (12 models, ~500GB total)"
echo "deepparallel download-models --count 12"
echo ""
echo "# Download without proprietary licensed models"
echo "deepparallel download-models --exclude llama3.1 gemma --count 8"
echo ""
echo "# Check model status"
echo "deepparallel model-status"
echo ""
echo "# Test the Genesis Engine with a simple question"
echo "deepparallel genesis 'What is the optimal temperature for protein folding?'"
echo ""
echo "# Run parallel reasoning"
echo "deepparallel reason 'Explain quantum entanglement' --paths 5"
echo ""
echo "# Run ensemble across multiple models"
echo "deepparallel ensemble 'Calculate the binding energy of water' --models gpt-4 claude-3"
echo ""

# Quick model availability check
echo "🔍 Checking model availability..."
echo ""

# Test if we can access HuggingFace
if command -v python3 &> /dev/null; then
    python3 -c "
import requests
try:
    response = requests.get('https://huggingface.co/api/models/meta-llama/Meta-Llama-3.1-8B-Instruct', timeout=5)
    if response.status_code == 200:
        print('✅ HuggingFace Hub accessible')
    else:
        print('⚠️  HuggingFace Hub may have access restrictions')
except:
    print('❌ Cannot reach HuggingFace Hub - check internet connection')
    "
fi

echo ""
echo "💡 Pro Tips:"
echo "• Start with a small ensemble (3-5 models) to test your setup"
echo "• Use --size-limit to control storage usage"
echo "• Configure cloud storage for model sharing across systems"
echo "• Monitor disk space - full ensemble requires ~500GB"
echo ""
echo "📖 For more details: deepparallel --help"
echo ""
