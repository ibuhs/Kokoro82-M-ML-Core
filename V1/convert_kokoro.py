import coremltools as ct
import torch
from kokoro import KPipeline
import numpy as np
import logging

def convert_text_encoder(text_encoder, save_path="text_encoder.mlpackage"):
    """Convert the text encoder component"""
    logging.info("Starting text encoder conversion...")
    
    # Print the text encoder structure
    logging.info("Detailed text encoder structure:")
    for name, module in text_encoder.named_children():
        logging.info(f"Layer: {name}")
        if isinstance(module, torch.nn.Embedding):
            logging.info(f"  Embedding dim: {module.embedding_dim}")
        elif isinstance(module, torch.nn.ModuleList):
            logging.info("  CNN layers:")
            for i, conv in enumerate(module):
                if isinstance(conv, torch.nn.Conv1d):
                    logging.info(f"    Conv{i}: in_channels={conv.in_channels}, "
                               f"out_channels={conv.out_channels}, "
                               f"kernel_size={conv.kernel_size}")
        elif isinstance(module, torch.nn.LSTM):
            logging.info(f"  LSTM input_size: {module.input_size}, hidden_size: {module.hidden_size}")
    
    try:
        # Create dummy inputs matching the expected shape
        batch_size = 1  # Force batch size to 1 for Core ML
        seq_length = 100
        
        # Create dummy input as int64 (for embedding layer)
        dummy_text = torch.randint(0, 100, (batch_size, seq_length), dtype=torch.long)
        
        # Create a wrapper class to handle the multiple inputs
        class TextEncoderWrapper(torch.nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
                
            def forward(self, text):
                return self.encoder.forward_without_packing(text)
        
        # Add forward_without_packing method to TextEncoder
        def forward_without_packing(self, text):
            x = text
            logging.info(f"Initial input shape: {x.shape}, dtype: {x.dtype}")
            
            if hasattr(self, 'embedding'):
                x = self.embedding(x)
                logging.info(f"Shape after embedding: {x.shape}, dtype: {x.dtype}")
            
            if hasattr(self, 'cnn'):
                x = x.transpose(1, 2)
                for conv in self.cnn:
                    x = conv(x)
                x = x.transpose(1, 2)
                logging.info(f"Shape after CNN: {x.shape}, dtype: {x.dtype}")
            
            if hasattr(self, 'lstm'):
                x, _ = self.lstm(x)
                logging.info(f"Shape after LSTM: {x.shape}, dtype: {x.dtype}")
            
            return x
        
        # Add the method to the text encoder
        import types
        text_encoder.forward_without_packing = types.MethodType(forward_without_packing, text_encoder)
        
        # Wrap the encoder and put in eval mode
        wrapped_encoder = TextEncoderWrapper(text_encoder).eval()
        
        logging.info("Successfully created wrapper")
        
        # Trace the model
        traced_model = torch.jit.trace(
            wrapped_encoder,
            dummy_text,
            check_trace=False,
            strict=False
        )
        logging.info("Successfully traced model")
        
        # Convert directly to Core ML
        model = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="input_text",
                    shape=(1, seq_length),  # Force batch size to 1
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
        
        model.save(save_path)
        logging.info(f"Saved model to {save_path}")
        
    except Exception as e:
        logging.error(f"Error in text encoder conversion: {str(e)}", exc_info=True)
        raise

def convert_style_encoder(style_encoder, save_path="style_encoder.mlpackage"):
    """Convert the style encoder component"""
    # Create dummy mel input
    dummy_mel = torch.randn(1, 80, 100)  # Adjust size based on actual input
    
    # Trace and convert
    traced_model = torch.jit.trace(style_encoder, dummy_mel)
    
    model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=dummy_mel.shape)],
        compute_units=ct.ComputeUnit.ALL
    )
    
    model.save(save_path)

def convert_decoder(decoder, save_path="decoder.mlpackage"):
    """Convert the decoder component"""
    # Create dummy inputs matching expected shapes
    dummy_text_hidden = torch.randn(1, 100, 512)  # Adjust dimensions
    dummy_style = torch.randn(1, 256)  # Adjust dimensions
    
    # Trace and convert
    traced_model = torch.jit.trace(
        decoder,
        (dummy_text_hidden, dummy_style)
    )
    
    model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(shape=dummy_text_hidden.shape),
            ct.TensorType(shape=dummy_style.shape)
        ],
        compute_units=ct.ComputeUnit.ALL
    )
    
    model.save(save_path)

# Main conversion function
def convert_kokoro_to_coreml():
    # Load Kokoro pipeline
    pipeline = KPipeline(lang_code='a')
    pipeline.model.eval()  # Put model in eval mode
    
    # Let's print the model structure
    logging.info("Model structure:")
    for name, module in pipeline.model.named_children():
        logging.info(f"Found component: {name}")
    
    # Convert only available components
    if hasattr(pipeline.model, 'text_encoder'):
        convert_text_encoder(pipeline.model.text_encoder)
    else:
        logging.warning("No text_encoder found in model")
        
    # We'll need to update these based on actual model structure
    # convert_style_encoder(pipeline.model.style_encoder)  # Remove this
    # convert_decoder(pipeline.model.decoder)  # Remove this 