import torch
import coremltools as ct
import numpy as np
from pathlib import Path
import sys
import os
from torch import nn
import logging
import types

# Add kokoro to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from kokoro import KModel
except ImportError:
    print("Could not import kokoro. Please make sure the kokoro module is in your Python path")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)

class CustomBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x):
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        x = (x - mean) / (var + 1e-5).sqrt()
        return self.gamma.view(1, -1, 1) * x + self.beta.view(1, -1, 1)

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(178, 512)  # Changed dimensions to match
        
        # Add CNN layers with custom BatchNorm
        self.cnn = nn.ModuleList([
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(512, 512, 5, padding=2)),
                CustomBatchNorm1d(512)
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(512, 512, 5, padding=2)),
                CustomBatchNorm1d(512)
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(512, 512, 5, padding=2)),
                CustomBatchNorm1d(512)
            )
        ])
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            batch_first=True,
            bidirectional=True  # This explains the _reverse weights
        )
        
    def forward_without_packing(self, text):
        x = text
        logging.info(f"Initial input shape: {x.shape}, dtype: {x.dtype}")
        
        # Embedding
        x = self.embedding(x)
        logging.info(f"After embedding: {x.shape}")
        
        # CNN layers
        x = x.transpose(1, 2)  # [B, E, T]
        for conv in self.cnn:
            x = conv(x)
        x = x.transpose(1, 2)  # [B, T, E]
        logging.info(f"After CNN: {x.shape}")
        
        # LSTM
        x, _ = self.lstm(x)
        logging.info(f"After LSTM: {x.shape}")
        
        return x
        
    def forward(self, x):
        return self.forward_without_packing(x)

class TextEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        
    def forward(self, text):
        return self.encoder.forward_without_packing(text)

class CustomLayerNorm(nn.Module):
    def __init__(self, num_features, hidden_size=128):
        super().__init__()
        self.fc = nn.Linear(hidden_size, num_features)
        self.num_features = num_features
        
    def forward(self, x):
        # x shape: [B, C, T]
        B, C, T = x.shape
        
        # Get conditioning vector
        cond = torch.zeros(B, T, self.fc.in_features, device=x.device)
        
        # Apply linear layer to get scale
        scale = self.fc(cond)  # [B, T, num_features]
        scale = scale.transpose(1, 2).contiguous()  # [B, num_features, T]
        
        # Compute normalization
        mean = x.mean(dim=1, keepdim=True)  # [B, 1, T]
        var = x.var(dim=1, keepdim=True, unbiased=False)  # [B, 1, T]
        x_norm = (x - mean) / (var + 1e-5).sqrt()  # [B, C, T]
        
        # Pad or slice scale to match x dimensions
        if scale.size(1) > x.size(1):
            scale = scale[:, :x.size(1), :]  # Slice to match
        elif scale.size(1) < x.size(1):
            padding = torch.ones_like(scale[:, :1, :]).expand(-1, x.size(1) - scale.size(1), -1)
            scale = torch.cat([scale, padding], dim=1)  # Pad to match
            
        # Apply scale
        out = x_norm * scale
        
        return out

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder part
        self.encode = nn.ModuleDict({
            'conv1': nn.utils.weight_norm(nn.Conv1d(514, 1024, 3, padding=1)),
            'conv2': nn.utils.weight_norm(nn.Conv1d(1024, 1024, 3, padding=1)),
            'norm1': CustomLayerNorm(1028, 128),
            'norm2': CustomLayerNorm(2048, 128),
            'conv1x1': nn.utils.parametrizations.weight_norm(nn.Conv1d(514, 1024, 1, bias=False))
        })
        
        # Decoder part
        self.decode = nn.ModuleList([
            nn.ModuleDict({
                'conv1': nn.utils.weight_norm(nn.Conv1d(1090, 1024, 3, padding=1)),
                'conv2': nn.utils.weight_norm(nn.Conv1d(1024, 1024, 3, padding=1)),
                'norm1': CustomLayerNorm(2180, 128),
                'norm2': CustomLayerNorm(2048, 128),
                'conv1x1': nn.utils.parametrizations.weight_norm(nn.Conv1d(1090, 1024, 1, bias=False))
            }) for _ in range(3)
        ])
        
        # Last decoder block
        self.decode.append(nn.ModuleDict({
            'conv1': nn.utils.weight_norm(nn.Conv1d(1090, 512, 3, padding=1)),
            'conv2': nn.utils.weight_norm(nn.Conv1d(512, 512, 3, padding=1)),
            'norm1': CustomLayerNorm(2180, 128),
            'norm2': CustomLayerNorm(1024, 128),
            'conv1x1': nn.utils.parametrizations.weight_norm(nn.Conv1d(1090, 512, 1, bias=False)),
            'pool': nn.utils.weight_norm(nn.Conv1d(1, 1090, 3, padding=1))
        }))
        
        # Additional components
        self.F0_conv = nn.utils.weight_norm(nn.Conv1d(1, 1, 3, padding=1))
        self.N_conv = nn.utils.weight_norm(nn.Conv1d(1, 1, 3, padding=1))
        self.asr_res = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(512, 64, 1))
        ])
        
        # Generator
        self.generator = nn.ModuleDict({
            'm_source': nn.ModuleDict({
                'l_linear': nn.Linear(9, 1)
            }),
            'noise_convs': nn.ModuleList([
                nn.Conv1d(22, 256, 12),
                nn.Conv1d(22, 128, 1)
            ]),
            'noise_res': self._make_noise_res_blocks(),
            'ups': nn.ModuleList([
                nn.utils.weight_norm(nn.ConvTranspose1d(512, 256, 20)),
                nn.utils.weight_norm(nn.ConvTranspose1d(256, 128, 12))
            ]),
            'resblocks': self._make_resblocks(),
            'conv_post': nn.utils.weight_norm(nn.Conv1d(128, 22, 7, padding=3))
        })
    
    def _make_noise_res_blocks(self):
        blocks = []
        configs = [(256, 7), (128, 11)]  # (channels, kernel_size)
        for channels, kernel_size in configs:
            block = {
                'convs1': nn.ModuleList([
                    nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2))
                    for _ in range(3)
                ]),
                'convs2': nn.ModuleList([
                    nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2))
                    for _ in range(3)
                ]),
                'adain1': nn.ModuleList([
                    CustomLayerNorm(channels*2, 128) for _ in range(3)
                ]),
                'adain2': nn.ModuleList([
                    CustomLayerNorm(channels*2, 128) for _ in range(3)
                ]),
                'alpha1': nn.ParameterList([
                    nn.Parameter(torch.ones(1, channels, 1)) for _ in range(3)
                ]),
                'alpha2': nn.ParameterList([
                    nn.Parameter(torch.ones(1, channels, 1)) for _ in range(3)
                ])
            }
            blocks.append(nn.ModuleDict(block))
        return nn.ModuleList(blocks)
    
    def _make_resblocks(self):
        kernel_sizes = [3, 7, 11, 3, 7, 11]
        blocks = []
        for kernel_size in kernel_sizes:
            channels = 128 if len(blocks) >= 3 else 256
            block = {
                'convs1': nn.ModuleList([
                    nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2))
                    for _ in range(3)
                ]),
                'convs2': nn.ModuleList([
                    nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2))
                    for _ in range(3)
                ]),
                'adain1': nn.ModuleList([
                    CustomLayerNorm(channels*2, 128) for _ in range(3)
                ]),
                'adain2': nn.ModuleList([
                    CustomLayerNorm(channels*2, 128) for _ in range(3)
                ]),
                'alpha1': nn.ParameterList([
                    nn.Parameter(torch.ones(1, channels, 1)) for _ in range(3)
                ]),
                'alpha2': nn.ParameterList([
                    nn.Parameter(torch.ones(1, channels, 1)) for _ in range(3)
                ])
            }
            blocks.append(nn.ModuleDict(block))
        return nn.ModuleList(blocks)
    
    def forward(self, text_features, voice_ref, speed):
        """Forward pass for the decoder.
        
        Args:
            text_features: [B, T, 512] text encoder output
            voice_ref: [B, 256] voice reference
            speed: [B] speed factor
        """
        # Debug input shapes
        logging.info(f"text_features shape: {text_features.shape}")
        logging.info(f"voice_ref shape: {voice_ref.shape}")
        logging.info(f"speed shape: {speed.shape}")
        
        # Encoder
        # Project voice_ref to 2 dimensions to match the conv1d input
        voice_ref = voice_ref[:, :2]  # Take first 2 dimensions [B, 2]
        voice_ref_expanded = voice_ref.unsqueeze(1).expand(-1, text_features.size(1), -1)
        x = torch.cat([text_features, voice_ref_expanded], dim=-1)  # [B, T, 512+2]
        logging.info(f"After concat shape: {x.shape}")
        
        x = x.transpose(1, 2)  # [B, 514, T]
        logging.info(f"After transpose shape: {x.shape}")
        
        # Apply encoder convolutions
        h1 = self.encode['conv1'](x)  # [B, 1024, T]
        h1 = self.encode['norm1'](h1)
        logging.info(f"After first conv shape: {h1.shape}")
        
        h2 = self.encode['conv2'](h1)  # [B, 1024, T]
        h2 = self.encode['norm2'](h2)
        logging.info(f"After second conv shape: {h2.shape}")
        
        h3 = self.encode['conv1x1'](x)  # [B, 1024, T]
        logging.info(f"After conv1x1 shape: {h3.shape}")
        
        x = h1 + h2 + h3  # [B, 1024, T]
        logging.info(f"After addition shape: {x.shape}")
        
        # Apply speed factor
        x = x * speed.view(-1, 1, 1)
        
        # Decoder blocks
        for i, block in enumerate(self.decode[:-1]):
            logging.info(f"Processing decoder block {i}")
            
            # Pad input to match expected channels (1090)
            padding = torch.zeros(x.size(0), 1090 - x.size(1), x.size(2), device=x.device)
            x_cond = torch.cat([x, padding], dim=1)  # [B, 1090, T]
            logging.info(f"After padding shape: {x_cond.shape}")
            
            h1 = block['conv1'](x_cond)  # [B, 1024, T]
            h1 = block['norm1'](h1)
            
            h2 = block['conv2'](h1)  # [B, 1024, T]
            h2 = block['norm2'](h2)
            
            h3 = block['conv1x1'](x_cond)  # [B, 1024, T]
            
            x = h1 + h2 + h3
            logging.info(f"After block {i}, shape: {x.shape}")
        
        # Last decoder block
        logging.info("Processing last decoder block")
        last_block = self.decode[-1]
        
        # Pad input to match expected channels (1090)
        padding = torch.zeros(x.size(0), 1090 - x.size(1), x.size(2), device=x.device)
        x_cond = torch.cat([x, padding], dim=1)  # [B, 1090, T]
        logging.info(f"After last padding shape: {x_cond.shape}")
        
        h1 = last_block['conv1'](x_cond)  # [B, 512, T]
        h1 = last_block['norm1'](h1)
        
        h2 = last_block['conv2'](h1)  # [B, 512, T]
        h2 = last_block['norm2'](h2)
        
        h3 = last_block['conv1x1'](x_cond)  # [B, 512, T]
        
        x = h1 + h2 + h3  # [B, 512, T]
        logging.info(f"After last block shape: {x.shape}")
        
        # Generator part
        # Project to correct dimensions before linear layer
        x = x.transpose(1, 2)  # [B, T, 512]
        
        # Fix the weight dimensions for the linear layer
        weight = self.generator['m_source']['l_linear'].weight  # [1, 9]
        weight = weight.transpose(0, 1)  # [9, 1]
        weight = weight.expand(-1, 512)  # [9, 512]
        bias = self.generator['m_source']['l_linear'].bias  # [1]
        
        x = torch.nn.functional.linear(x, weight)  # [B, T, 9]
        if bias is not None:
            x = x + bias
        
        logging.info(f"After m_source linear shape: {x.shape}")
        
        # Additional processing
        x = x.transpose(1, 2)  # [B, 9, T]
        
        return x

def convert_text_encoder(model_path, output_path):
    """Convert text encoder model to Core ML format."""
    logging.info("Starting text encoder conversion...")
    
    # Create model instance
    model = TextEncoder()
    
    # Load weights
    state_dict = torch.load(model_path, weights_only=False)
    
    # Debug print
    print("\nState dict keys:")
    for key in sorted(state_dict.keys()):
        print(f"  {key}: {state_dict[key].shape}")
    
    print("\nModel state dict keys:")
    model_state = model.state_dict()
    for key in sorted(model_state.keys()):
        print(f"  {key}: {model_state[key].shape}")
    
    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create wrapper
    wrapped_model = TextEncoderWrapper(model)
    
    # Create dummy input
    dummy_text = torch.randint(0, 100, (1, 100), dtype=torch.long)
    
    # Trace the model
    traced_model = torch.jit.trace(
        wrapped_model,
        dummy_text,
        check_trace=False,
        strict=False
    )
    logging.info("Successfully traced model")
    
    # Convert to Core ML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="input_text",
                shape=(1, 100),  # batch_size, sequence_length
                dtype=np.int32
            )
        ],
        outputs=[
            ct.TensorType(
                name="output",
                dtype=np.float32
            )
        ],
        source="pytorch",
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS15
    )
    logging.info("Successfully converted to Core ML")
    
    # Save the model
    mlmodel.save(output_path)
    logging.info(f"Saved model to {output_path}")

def convert_decoder(model_path, output_path):
    """Convert decoder model to Core ML format."""
    logging.info("Starting decoder conversion...")
    
    # Create model instance
    model = Decoder()
    model.eval()  # Important: set to eval mode before loading weights
    
    # Load weights
    state_dict = torch.load(model_path, weights_only=False)
    model.load_state_dict(state_dict)
    
    # Create example inputs with correct dimensions
    batch_size = 1
    seq_len = 100
    example_text_features = torch.randn(batch_size, seq_len, 512)  # [B, T, 512]
    example_voice_ref = torch.randn(batch_size, 256)  # [B, 256]
    example_speed = torch.ones(batch_size)  # [B]
    
    # Add debug logging
    logging.info(f"Input shapes:")
    logging.info(f"  text_features: {example_text_features.shape}")
    logging.info(f"  voice_ref: {example_voice_ref.shape}")
    logging.info(f"  speed: {example_speed.shape}")
    
    # Try a forward pass before tracing
    with torch.no_grad():
        try:
            output = model(example_text_features, example_voice_ref, example_speed)
            logging.info(f"Test forward pass output shape: {output.shape}")
        except Exception as e:
            logging.error(f"Forward pass failed: {e}")
            logging.error("Stack trace:", exc_info=True)
            raise
    
    # Trace the model
    traced_model = torch.jit.trace(
        model,
        (example_text_features, example_voice_ref, example_speed),
        check_trace=False
    )
    
    # Convert to Core ML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="text_features",
                shape=(1, 100, 512)  # Fixed batch and sequence length
            ),
            ct.TensorType(
                name="voice_ref",
                shape=(1, 256)
            ),
            ct.TensorType(
                name="speed",
                shape=(1,)
            )
        ],
        outputs=[
            ct.TensorType(
                name="audio"
                # Let Core ML infer the shape
            )
        ],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS15
    )
    
    mlmodel.save(output_path)
    logging.info(f"Saved model to {output_path}")

def save_models():
    """Extract and save models from KModel."""
    # Initialize model
    model = KModel().to('cpu').eval()
    
    # Save text encoder state dict
    torch.save(model.text_encoder.state_dict(), "models/text_encoder.pt")
    print("Saved text encoder state dict")
    
    # Save decoder state dict
    torch.save(model.decoder.state_dict(), "models/decoder.pt")
    print("Saved decoder state dict")
    
    # Print model structure for debugging
    print("\nModel structure:")
    print("Text Encoder:")
    for name, param in model.text_encoder.named_parameters():
        print(f"  {name}: {param.shape}")
    
    print("\nDecoder:")
    for name, param in model.decoder.named_parameters():
        print(f"  {name}: {param.shape}")

def main():
    # Create directories
    Path("models").mkdir(exist_ok=True)
    output_dir = Path("Resources/Models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # First, save the models from KModel
    print("Extracting models from KModel...")
    save_models()
    
    # Then convert them to Core ML
    print("\nConverting to Core ML format...")
    convert_text_encoder(
        "models/text_encoder.pt",
        output_dir / "text_encoder.mlpackage"
    )
    
    convert_decoder(
        "models/decoder.pt",
        output_dir / "decoder.mlpackage"
    )

if __name__ == "__main__":
    main() 