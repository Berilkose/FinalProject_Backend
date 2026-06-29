import os
import threading
from flask import Flask, request, jsonify
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
import io
import torch
import google.generativeai as genai
from transformers import AutoModelForImageClassification, AutoImageProcessor

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
db = SQLAlchemy(app)
migrate = Migrate(app, db) # Migrasyonları yönetir

# --- AI Modelleri (Uygulama başladığında 1 kez yüklenir) ---
MODEL_NAME = "dima806/house-plant-image-detection"

model = None
processor = None

def get_model():
    global model, processor
    if model is None:
        print("Model ilk kez yükleniyor...")
        # Modeli burada yüklüyoruz
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        print("Model başarıyla yüklendi!")
    return model, processor
# API yapılandırması
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model_gemini = genai.GenerativeModel('gemini-1.5-flash')

# --- Veritabanı Modelleri ---
class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    date = db.Column(db.String(10)) # "YYYY-MM-DD" formatında saklayacağız
    done = db.Column(db.Boolean, default=False)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_message = db.Column(db.String(500))
    bot_reply = db.Column(db.String(500))

# --- Rotalar ---
@app.route('/tasks', methods=['POST'])
def add_task():
    data = request.json
    # Eğer birden fazla tarih geldiyse (Weekly için) döngüye sok
    dates = data.get('dates', [data.get('date')]) 
    for d in dates:
        new_task = Task(name=data['name'], date=d)
        db.session.add(new_task)
    db.session.commit()
    return jsonify({"message": "Success"}), 201

@app.route('/tasks/<int:id>', methods=['DELETE'])
def delete_task(id):
    task = Task.query.get_or_404(id)
    db.session.delete(task)
    db.session.commit()
    return jsonify({"message": "Deleted"})

@app.route('/upload', methods=['POST'])
def upload_plant():
    # Modeli burada çağırıyoruz
    current_model, current_processor = get_model()
    
    if 'file' not in request.files:
        return jsonify({"error": "File cannot be found."}), 400
    
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    inputs = current_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = current_model(**inputs)
        predicted_class_idx = outputs.logits.argmax(-1).item()

    predicted_label = model.config.id2label[predicted_class_idx]
    return jsonify({
        "plant_name": predicted_label,
        "care_tips": "For detailed care guide, please ask the chatbot."
    })

@app.route('/tasks', methods=['GET'])
def get_tasks():
    date_filter = request.args.get('date') # Flutter'dan gelen ?date=... kısmını okur
    if date_filter:
        tasks = Task.query.filter_by(date=date_filter).all()
    else:
        tasks = Task.query.all()
        
    return jsonify([{"id": t.id, "name": t.name, "date": t.date, "done": t.done} for t in tasks])

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    
    # Gemini'den cevap al
    response = model_gemini.generate_content(user_message)
    reply = response.text
    
    # Geçmişi kaydet
    history = ChatHistory(user_message=user_message, bot_reply=reply)
    db.session.add(history)
    db.session.commit()
    return jsonify({"reply": reply})

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)