from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import os
import bcrypt
import jwt
from datetime import datetime, timedelta
from pymongo import MongoClient
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
import torch.optim as optim

# ====== CONFIG ======
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

JWT_SECRET = os.environ.get("JWT_SECRET")
MONGO_URI = os.environ.get("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client['authdb']
users = db['users']

# ====== CNN MODEL SETUP ======
class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 16 * 16)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

classes = [
    "Tomato_Bacterial_Spot", "Tomato_Early_Blight", "Tomato_Leaf_Mold",
    "Tomato_Septoria_Spot", "Tomato_Yellow_Leaf_Curl", "Tomato_Healthy",
    "Potato_Early_Blight", "Potato_Late_Blight", "Potato_Healthy",
    "Corn_Common_Rust"
]

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = FakeData(size=100, image_size=(3, 64, 64), num_classes=len(classes), transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cpu")
model = TinyCNN(num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

model.eval()

disease_info = {
    "Tomato_Bacterial_Spot": {"definition": "Bacterial spots on leaves and fruits.", "color": "red", "health_status": "Unhealthy"},
    "Tomato_Early_Blight": {"definition": "Fungal dark spots on older leaves.", "color": "brown", "health_status": "Unhealthy"},
    "Tomato_Leaf_Mold": {"definition": "Yellow spots with mold under leaves.", "color": "orange", "health_status": "Unhealthy"},
    "Tomato_Septoria_Spot": {"definition": "Gray-centered circular spots.", "color": "gray", "health_status": "Unhealthy"},
    "Tomato_Yellow_Leaf_Curl": {"definition": "Curling and yellowing of leaves.", "color": "yellow", "health_status": "Unhealthy"},
    "Tomato_Healthy": {"definition": "Healthy tomato leaf.", "color": "green", "health_status": "Healthy"},
    "Potato_Early_Blight": {"definition": "Brown spots with rings on leaves.", "color": "brown", "health_status": "Unhealthy"},
    "Potato_Late_Blight": {"definition": "Rapid lesions on leaves and stems.", "color": "darkred", "health_status": "Unhealthy"},
    "Potato_Healthy": {"definition": "Healthy potato foliage.", "color": "green", "health_status": "Healthy"},
    "Corn_Common_Rust": {"definition": "Red-brown pustules on corn leaves.", "color": "red", "health_status": "Unhealthy"}
}

# ====== ML PREDICTION ROUTE ======
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    image = Image.open(filepath).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = classes[predicted.item()]

    info = disease_info.get(predicted_class, {
        'definition': 'Unknown disease.',
        'color': 'gray',
        'health_status': 'Unknown'
    })

    return jsonify({
        'result': predicted_class,
        'definition': info['definition'],
        'color': info['color'],
        'healthy': info['health_status'] == "Healthy",
        'image_path': filepath
    })

# ====== AUTH ROUTES ======

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if users.find_one({"email": data['email']}):
        return jsonify({'error': 'Email already exists'}), 400

    hashed = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())
    user = {
        "name": data['name'],
        "email": data['email'],
        "password": hashed,
        "q1": data['q1'],
        "q2": data['q2'],
        "createdAt": datetime.utcnow()
    }
    users.insert_one(user)
    return jsonify({'message': 'Registered successfully'})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = users.find_one({"email": data['email']})
    if not user or not bcrypt.checkpw(data['password'].encode(), user['password']):
        return jsonify({'error': 'Invalid email or password'}), 401

    token = jwt.encode({
        "userId": str(user['_id']),
        "name": user['name'],
        "exp": datetime.utcnow() + timedelta(hours=2)
    }, JWT_SECRET, algorithm="HS256")

    return jsonify({'message': 'Logged in', 'token': token, 'name': user['name']})

@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    data = request.get_json()
    user = users.find_one({"email": data['email']})
    if not user:
        return jsonify({'error': 'User not found'}), 404

    if user['q1'] != data['q1'] or user['q2'] != data['q2']:
        return jsonify({'error': 'Security answers did not match'}), 401

    new_hashed = bcrypt.hashpw(data['newPass'].encode(), bcrypt.gensalt())
    users.update_one({"_id": user['_id']}, {"$set": {"password": new_hashed}})
    return jsonify({'message': 'Password updated'})

@app.route('/delete', methods=['DELETE'])
def delete_account():
    data = request.get_json()
    user = users.find_one({"email": data['email']})
    if not user or not bcrypt.checkpw(data['password'].encode(), user['password']):
        return jsonify({'error': 'Invalid email or password'}), 401

    users.delete_one({"_id": user['_id']})
    return jsonify({'message': 'Account deleted'})

@app.route('/users', methods=['GET'])
def get_users():
    user_list = list(users.find({}, {"password": 0}))
    for user in user_list:
        user['_id'] = str(user['_id'])
    return jsonify({'count': len(user_list), 'users': user_list})

@app.route('/')
def index():
    return render_template('index.html')

# ====== START APP ======
if __name__ == '__main__':
    app.run(debug=True)
