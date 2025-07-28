#!/usr/bin/env python3
"""
Model Download Script for VIVID Project
Downloads and sets up all required pre-trained models
"""

import os
import torch
import urllib.request
from pathlib import Path
import sys

def create_directories():
    """Create necessary directories for models (optional, models download to default locations)"""
    # Models will be downloaded to default cache locations:
    # - YOLO: ~/.ultralytics/ 
    # - MiDaS: torch hub cache
    # - DeepSORT: current directory if needed
    print("✓ Models will be downloaded to default cache locations")

def download_yolo_models():
    """Download YOLO v8 models to default ultralytics cache"""
    print("\n📥 Downloading YOLO v8 models...")
    
    try:
        from ultralytics import YOLO
        
        # Download the specific model used in your code
        print("  Downloading yolov8n.pt (used in vivid.py)...")
        model = YOLO("yolov8n.pt")  # This matches your code exactly
        print("  ✓ yolov8n.pt downloaded successfully")
        
        # Optional: Download other variants for user choice
        other_models = {
            "yolov8s.pt": "Small - Better accuracy than nano",
            "yolov8m.pt": "Medium - Even better accuracy"
        }
        
        print("\n  Optional models (you can switch to these in vivid.py):")
        for model_name, description in other_models.items():
            try:
                print(f"    Downloading {model_name} ({description})...")
                YOLO(model_name)
                print(f"    ✓ {model_name} ready")
            except Exception as e:
                print(f"    ⚠️ Failed to download {model_name}: {e}")
                
    except ImportError:
        print("  ⚠️ ultralytics not installed. Install with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"  ⚠️ Error downloading YOLO models: {e}")
        return False
    
    return True

def download_midas_models():
    """Download MiDaS depth estimation models to torch hub cache"""
    print("\n📥 Downloading MiDaS models...")
    
    try:
        # Download the specific model used in your code
        print("  Loading DPT_Large (used in vivid.py)...")
        model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True)
        transform = torch.hub.load("intel-isl/MiDaS", "transforms")
        print("  ✓ DPT_Large and transforms loaded and cached")
        
        del model  # Free memory
        
        # Optional: Download other variants
        print("\n  Optional MiDaS models:")
        other_models = ["DPT_Hybrid", "MiDaS_small"]
        
        for model_name in other_models:
            try:
                print(f"    Loading {model_name}...")
                model = torch.hub.load("intel-isl/MiDaS", model_name, pretrained=True)
                print(f"    ✓ {model_name} cached")
                del model
            except Exception as e:
                print(f"    ⚠️ Failed to load {model_name}: {e}")
                
    except Exception as e:
        print(f"  ⚠️ Error downloading MiDaS models: {e}")
        print("     Models will be downloaded on first run of vivid.py")
        return False
    
    return True

def download_deepsort_weights():
    """Download DeepSORT tracking model weights if needed"""
    print("\n📥 Checking DeepSORT setup...")
    
    # Check if your code uses external DeepSORT weights
    # If DeepSORT is integrated differently, this can be skipped
    print("  DeepSORT tracking uses built-in feature extraction")
    print("  ✓ No additional downloads needed for tracking")
    
    # If you do use external weights, uncomment below:
    # weights_url = "https://github.com/ZQPei/deep_sort_pytorch/releases/download/v1.0/ckpt.t7"
    # weights_path = "ckpt.t7"
    # 
    # try:
    #     if not os.path.exists(weights_path):
    #         print(f"  Downloading DeepSORT weights...")
    #         urllib.request.urlretrieve(weights_url, weights_path)
    #         print(f"  ✓ DeepSORT weights saved to {weights_path}")
    #     else:
    #         print(f"  ✓ DeepSORT weights already exist")
    # except Exception as e:
    #     print(f"  ⚠️ Failed to download DeepSORT weights: {e}")
    
    return True

def verify_installations():
    """Verify that all required packages are installed"""
    print("\n🔍 Verifying installations...")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV (cv2)',
        'ultralytics': 'Ultralytics YOLO',
        'numpy': 'NumPy',
        'PyQt5': 'PyQt5'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PyQt5':
                import PyQt5
            else:
                __import__(package)
            print(f"  ✓ {name} installed")
        except ImportError:
            print(f"  ❌ {name} NOT installed")
            missing_packages.append(name)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("   Please install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def check_cuda_availability():
    """Check CUDA availability for GPU acceleration"""
    print("\n🔧 Checking CUDA availability...")
    
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available")
        print(f"  ✓ GPU Device: {torch.cuda.get_device_name()}")
        print(f"  ✓ CUDA Version: {torch.version.cuda}")
        print(f"  ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ⚠️ CUDA not available - will use CPU")
        print("     Performance may be slower without GPU acceleration")

def main():
    """Main function to download all models"""
    print("🚀 VIVID Model Download Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("vivid.py") and not os.path.exists("app.py"):
        print("⚠️ Warning: Run this script from the VIVID project root directory")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Verify installations first
    if not verify_installations():
        print("\n❌ Please install missing packages before downloading models")
        sys.exit(1)
    
    # Check CUDA
    check_cuda_availability()
    
    # Create directory structure
    create_directories()
    
    # Download models
    download_yolo_models()
    download_midas_models()
    download_deepsort_weights()
    
    print("\n" + "=" * 50)
    print("🎉 Model download complete!")
    print("\nYour models are now cached and ready:")
    print("• YOLO v8: ~/.ultralytics/")
    print("• MiDaS: torch hub cache")
    print("\nNext steps:")
    print("1. Run the main application: python app.py")
    print("2. Or run with specific camera: python app.py --source 1")
    print("3. See README.md for detailed usage instructions")
    
    # Show cache locations
    try:
        import torch
        print(f"\n📁 Torch hub cache: {torch.hub.get_dir()}")
    except:
        pass

if __name__ == "__main__":
    main()
