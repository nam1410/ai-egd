#!/usr/bin/env python
import os
import timm
import torch

# Create a cache directory
cache_dir = os.path.expanduser("~/ai_egd_models")
os.makedirs(cache_dir, exist_ok=True)

# Set environment variable for timm cache
os.environ['TIMM_MODEL_DIR'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(cache_dir, 'hub')

print(f"Downloading models to: {cache_dir}")

# Download the models
models_to_download = [
    'vit_small_patch14_dinov2.lvd142m',
    'vit_base_patch14_dinov2.lvd142m',
    'vit_large_patch14_dinov2.lvd142m'
]

for model_name in models_to_download:
    print(f"\nDownloading {model_name}...")
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=2)
        print(f"✓ Successfully downloaded {model_name}")
        # Get model info
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")

print(f"\nModels downloaded to: {cache_dir}")
print("You can now use these models offline by setting TIMM_MODEL_DIR environment variable")