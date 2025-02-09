import torch
import numpy as np
from pathlib import Path

def extract_voice_reference(pt_file):
    """Extract reference data from a voice .pt file."""
    # Load the PyTorch file
    voice_data = torch.load(pt_file, map_location='cpu')
    
    print(f"Voice data type: {type(voice_data)}")
    
    if isinstance(voice_data, torch.Tensor):
        print(f"Tensor shape: {voice_data.shape}")
        print(f"Tensor type: {voice_data.dtype}")
        
        # Convert to numpy
        ref_style = voice_data.numpy()
        
        # The tensor is [510, 1, 256] - we want to get a single style vector
        # Take mean over the first dimension (510 frames)
        ref_style = np.mean(ref_style, axis=0)
        print(f"Shape after time averaging: {ref_style.shape}")
        
        # Squeeze out the middle dimension (batch)
        ref_style = np.squeeze(ref_style)
        print(f"Shape after squeezing: {ref_style.shape}")
        
        # Add batch dimension back to get [1, 256]
        ref_style = ref_style.reshape(1, -1)
        print(f"Final shape: {ref_style.shape}")
        
        return ref_style.astype(np.float32)
    else:
        raise ValueError(f"Expected torch.Tensor, got {type(voice_data)}")

def main():
    voice_dir = Path("voices")
    output_dir = Path("Resources/Voices")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each .pt file
    for pt_file in voice_dir.glob("*.pt"):
        try:
            print(f"\nProcessing {pt_file.name}...")
            ref_data = extract_voice_reference(pt_file)
            
            # Save as binary file
            output_file = output_dir / f"{pt_file.stem}.data"
            ref_data.tofile(output_file)
            
            print(f"Saved to {output_file.name}")
            print(f"Shape: {ref_data.shape}, Size: {ref_data.nbytes} bytes")
            
        except Exception as e:
            print(f"Error processing {pt_file.name}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 