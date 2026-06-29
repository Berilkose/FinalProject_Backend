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
model = None
processor = None

def load_model():
    global model, processor
    if model is None:
        print("Model yükleniyor...")
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        print("Model yüklendi!")

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
@app.route('/tasks', methods=['GET'])
def get_tasks():
    # Test için basit bir yanıt
    return jsonify([{"id": 1, "name": "Water the Plants"}])

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    reply = bot.chat_reply(user_message)
    return jsonify({"reply": reply})

@app.route('/upload', methods=['POST'])
def upload_plant():
    load_model()

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class_idx = outputs.logits.argmax(-1).item()

    predicted_label = model.config.id2label[predicted_class_idx]

    return jsonify({
        "plant_name": predicted_label,
        "care_tips": f"You can ask the chatbot for detailed maintenance information."
    })

# bir kerelik çalışsa yeterli:
with app.app_context():
    db.create_all() # Bu komut tablolarını veritabanında otomatik oluşturur.

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)