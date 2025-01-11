import os

from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

loaded_model = None
loaded_scaler = None
loaded_label_encoder = None


def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


@app.before_request
def startup_event():
    global loaded_model
    global loaded_scaler
    global loaded_label_encoder
    if loaded_model is None:
        loaded_model = load_model('/home/fuxkinghatred/diary_of_emotions/model_k_nearest_neighbors.pkl')
    if loaded_scaler is None:
        loaded_scaler = load_model('/home/fuxkinghatred/diary_of_emotions/scaler.pkl')
    if loaded_label_encoder is None:
        loaded_label_encoder = load_model('/home/fuxkinghatred/diary_of_emotions/label_encoder.pkl')


@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    data = request.get_json(force=True)
    if not all(key in data for key in ['h', 's', 'l']):
        return jsonify({"error": "Недостаточно данных. Пожалуйста, предоставьте значения для 'h', 's' и 'l'."}), 400
    h = float(data["h"])
    s = float(data["s"])
    l = float(data["l"])
    if not (0 <= h <= 360 and 0 <= s <= 100 and 0 <= l <= 100):
        return jsonify({
                           "error": "Данные вне границ. Пожалуйста, предоставьте значения для 'h' от 0 до 360, для 's' и 'l' от 0 до 100."}), 400
    input_vector = np.array([[h, s, l]])
    scaled_input = loaded_scaler.transform(input_vector)
    prediction = loaded_model.predict(scaled_input)[0]
    predicted_emotion = loaded_label_encoder.inverse_transform([prediction])[0]
    return jsonify({"predicted_emotion": predicted_emotion})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
