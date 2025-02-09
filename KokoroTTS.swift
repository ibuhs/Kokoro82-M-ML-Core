import CoreML
import AVFoundation

class KokoroTTS {
    private let textEncoder: MLModel
    private let styleEncoder: MLModel
    private let decoder: MLModel
    
    init() throws {
        // Load the Core ML models
        guard let textEncoderURL = Bundle.main.url(forResource: "text_encoder", withExtension: "mlmodelc"),
              let styleEncoderURL = Bundle.main.url(forResource: "style_encoder", withExtension: "mlmodelc"),
              let decoderURL = Bundle.main.url(forResource: "decoder", withExtension: "mlmodelc") else {
            throw NSError(domain: "KokoroTTS", code: -1, userInfo: [NSLocalizedDescriptionKey: "Missing model files"])
        }
        
        textEncoder = try MLModel(contentsOf: textEncoderURL)
        styleEncoder = try MLModel(contentsOf: styleEncoderURL)
        decoder = try MLModel(contentsOf: decoderURL)
    }
    
    func synthesize(text: String) throws -> AVAudioPCMBuffer {
        // Create input dictionary for text encoder
        let textFeature = try MLFeatureValue(
            multiArray: text.toInputArray()  // You'll need to implement this helper
        )
        let textInput = try MLDictionary(dictionary: ["input": textFeature])
        
        // Run text encoder
        let textEncoderOutput = try textEncoder.prediction(from: textInput)
        let textHidden = textEncoderOutput.featureValue(for: "output")!
        
        // Create style embedding (you may want to cache this)
        let styleInput = try MLDictionary(dictionary: [
            "input": MLFeatureValue(multiArray: createDefaultStyle())  // Implement this
        ])
        let styleOutput = try styleEncoder.prediction(from: styleInput)
        let style = styleOutput.featureValue(for: "output")!
        
        // Run decoder
        let decoderInput = try MLDictionary(dictionary: [
            "text_hidden": textHidden,
            "style": style
        ])
        let decoderOutput = try decoder.prediction(from: decoderInput)
        
        // Convert output to audio buffer
        return try createAudioBuffer(from: decoderOutput.featureValue(for: "output")!)
    }
} 