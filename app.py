from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from transformers import pipeline
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import io
import requests
import os

app = Flask(__name__)
# Modelin yolu
MODEL_NAME = "dima806/house-plant-image-detection"

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') 
db = SQLAlchemy(app)

# Modeli ve işlemciyi global yükle
print("Model yükleniyor...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
print("Model başarıyla yüklendi!")

class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    # ... diğer alanlar ...

# Chatbot Modeli Yükleme
class ChatBot:
    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.model = "mistral"

    def chat_reply(self, message):
        r = requests.post(self.url, json={
            "model": self.model,
            "prompt": message,
            "stream": False
        })
        return r.json()["response"]
    
bot = ChatBot()

# --- ROUTES ---

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    reply = bot.chat_reply(user_message)
    return jsonify({"reply": reply})

@app.route('/upload', methods=['POST'])
def upload_plant():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    
    # Görüntüyü modele hazırla
    inputs = processor(images=image, return_tensors="pt")
    
    # Tahmin yap
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
    
    # İsimleri config.json'dan otomatik alabiliriz (label2id/id2label)
    # Burada senin verdiğin id2label listesi üzerinden eşleşme yapıyoruz
    predicted_label = model.config.id2label[predicted_class_idx]
    
    return jsonify({
        "plant_name": predicted_label,
        "care_tips": f"Plant name {predicted_label}. You can ask the chatbot for detailed maintenance information."
    })

# bir kerelik çalışsa yeterli:
with app.app_context():
    db.create_all() # Bu komut tablolarını veritabanında otomatik oluşturur.

if __name__ == '__main__':
    app.run()