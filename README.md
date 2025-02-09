# Kokoro TTS Core ML Implementation

This repository contains the tools to convert a Kokoro TTS model to Core ML format and instructions for implementing it in an iOS app. I have no way to test or confirm if this works. 


## Converting the Model

1. Install required Python packages:

```bash
pip install torch coremltools kokoro numpy
```

2. Run the conversion script:

```bash
python run_conversion.py
```

This will generate `text_encoder.mlpackage`, which can be used in your iOS app.

## iOS Implementation

### 1. Add Model to Xcode Project

1. Drag `text_encoder.mlpackage` into your Xcode project.
2. Make sure "Copy items if needed" is checked.
3. Add to your target.

### 2. Basic Implementation

```swift
import CoreML

class KokoroTTSManager {
    private let textEncoder: text_encoder
    
    init() throws {
        // Initialize the model
        guard let modelURL = Bundle.main.url(forResource: "text_encoder", withExtension: "mlmodelc") else {
            throw KokoroError.modelNotFound
        }
        self.textEncoder = try text_encoder(contentsOf: modelURL)
    }
    
    func generateSpeech(from text: String) async throws -> [Float] {
        // Convert text to input tensor
        let tokenizedInput = tokenizeText(text)
        
        // Create MLMultiArray input
        let inputShape: [NSNumber] = [1, 100] // Batch size 1, sequence length 100
        let input = try MLMultiArray(shape: inputShape, dataType: .int32)
        
        // Fill input with tokenized text
        for (i, token) in tokenizedInput.enumerated() {
            if i < 100 { // Ensure we don't exceed max length
                input[i] = NSNumber(value: token)
            }
        }
        
        // Create model input
        let modelInput = text_encoderInput(input_text: input)
        
        // Get prediction
        let output = try await textEncoder.prediction(input: modelInput)
        
        // Process output
        return convertOutputToArray(output.output)
    }
    
    private func tokenizeText(text: String) -> [Int32] {
        // Implement your text tokenization here
        // This should match your Python tokenization
        return []
    }
    
    private func convertOutputToArray(output: MLMultiArray) -> [Float] {
        // Convert MLMultiArray to Float array
        let length = output.count
        var result = [Float](repeating: 0, count: length)
        for i in 0..<length {
            result[i] = output[i].floatValue
        }
        return result
    }
}

enum KokoroError: Error {
    case modelNotFound
    case invalidInput
    case processingError
}
```

### 3. Usage Example

```swift
class ViewController: UIViewController {
    private var ttsManager: KokoroTTSManager?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        do {
            ttsManager = try KokoroTTSManager()
        } catch {
            print("Failed to initialize TTS: \(error)")
        }
    }
    
    @IBAction func generateSpeech(_ sender: Any) {
        Task {
            do {
                let text = "Hello, world!"
                let output = try await ttsManager?.generateSpeech(from: text)
                // Process output...
            } catch {
                print("Speech generation failed: \(error)" )
            }
        }
    }
}
```

## Model Details

### Text Encoder
- **Input Shape**: `[1, 100]` (batch_size, sequence_length)
- **Input Type**: `Int32`
- **Output Shape**: `[1, 100, 512]` (batch_size, sequence_length, hidden_size)
- **Output Type**: `Float32`

### Input Preparation
1. Tokenize input text to integers.
2. Pad or truncate to length 100.
3. Convert to MLMultiArray.

### Output Processing
The output is a feature representation that can be used for:
- Style transfer
- Voice conversion
- Direct audio generation

## Performance Considerations
1. The model is optimized for batch size 1.
2. Maximum sequence length is 100 tokens.
3. Uses **ALL** compute units for best performance.
4. Minimum deployment target is **iOS 15**.


