import torch
import numpy as np
from pathlib import Path

def inspect_voice_file(pt_file):
    print(f"\nInspecting {pt_file.name}...")
    
    try:
        # Load the PyTorch file
        voice_data = torch.load(pt_file, map_location='cpu')
        
        # Print the type of the loaded data
        print(f"\nLoaded data type: {type(voice_data)}")
        
        if isinstance(voice_data, dict):
            # Print all keys in the dictionary
            print("\nKeys in voice file:", voice_data.keys())
            
            # Print info for each item in the dictionary
            for key, value in voice_data.items():
                print(f"\nKey: {key}")
                print(f"Type: {type(value)}")
                if isinstance(value, torch.Tensor):
                    print(f"Shape: {value.shape}")
                    print(f"Dtype: {value.dtype}")
                    print(f"Device: {value.device}")
                    print(f"Requires grad: {value.requires_grad}")
                    # Try to access the tensor data safely
                    try:
                        if value.dim() == 3:
                            print("3D tensor found!")
                            print(f"Dimensions: {value.size(0)} x {value.size(1)} x {value.size(2)}")
                            # Try to get mean across time dimension
                            mean_value = value.mean(dim=1)
                            print(f"Mean shape: {mean_value.shape}")
                    except Exception as e:
                        print(f"Error accessing tensor: {str(e)}")
        else:
            # If it's a tensor directly
            if isinstance(voice_data, torch.Tensor):
                print(f"Direct tensor:")
                print(f"Shape: {voice_data.shape}")
                print(f"Dtype: {voice_data.dtype}")
                print(f"Device: {voice_data.device}")
                print(f"Requires grad: {voice_data.requires_grad}")
            else:
                print("Voice file contains unknown type")
                print(f"Content type: {type(voice_data)}")
    
    except Exception as e:
        print(f"Error inspecting file: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    voice_dir = Path("voices")
    if not voice_dir.exists():
        print(f"Directory {voice_dir} not found!")
        return
        
    voice_files = list(voice_dir.glob("*.pt"))
    if not voice_files:
        print("No .pt files found in 'voices' directory")
        return
    
    # Check first few files
    for file in voice_files[:3]:  # Just check first 3 files
        try:
            inspect_voice_file(file)
        except Exception as e:
            print(f"Error inspecting {file.name}:")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 