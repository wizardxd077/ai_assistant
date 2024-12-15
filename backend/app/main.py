import os
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import (
    BartForConditionalGeneration, BartTokenizer,
    ViTForImageClassification, ViTImageProcessor,
    Wav2Vec2ForCTC, Wav2Vec2Processor
)
import cv2
import librosa
import speech_recognition as sr

class MultimodalAIAssistant:
    def __init__(self):
        # Initialize models
        self.nlp_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        self.nlp_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
        
        self.vision_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.vision_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.audio_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")
        
        # Speech recognition
        self.recognizer = sr.Recognizer()

    def process_text(self, text):
        inputs = self.nlp_tokenizer(text, return_tensors="pt")
        summary_ids = self.nlp_model.generate(inputs["input_ids"])
        return self.nlp_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.vision_processor(images=image, return_tensors="pt")
        outputs = self.vision_model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        return self.vision_model.config.id2label[predicted_class_idx]

    def process_audio(self, audio_path):
        waveform, rate = librosa.load(audio_path, sr=16000)
        input_values = self.audio_processor(waveform, return_tensors="pt").input_values
        logits = self.audio_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.audio_processor.batch_decode(predicted_ids)[0]
        return transcription

app = Flask(__name__)
CORS(app)
assistant = MultimodalAIAssistant()

@app.route('/process/text', methods=['POST'])
def process_text():
    data = request.json
    result = assistant.process_text(data['text'])
    return jsonify({"response": result})

@app.route('/process/image', methods=['POST'])
def process_image():
    file = request.files['image']
    filename = os.path.join('uploads', file.filename)
    file.save(filename)
    result = assistant.process_image(filename)
    return jsonify({"classification": result})

@app.route('/process/audio', methods=['POST'])
def process_audio():
    file = request.files['audio']
    filename = os.path.join('uploads', file.filename)
    file.save(filename)
    result = assistant.process_audio(filename)
    return jsonify({"transcription": result})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=5000)
