from convert_kokoro import convert_kokoro_to_coreml
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    try:
        # This will create text_encoder.mlpackage, style_encoder.mlpackage, and decoder.mlpackage
        convert_kokoro_to_coreml()
        print("Conversion completed successfully!")
    except Exception as e:
        logging.error(f"Conversion failed: {str(e)}", exc_info=True) 