import numpy as np
from flask import Flask, jsonify, request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = int(model.predict([[data['exp']]])[0])
    output = prediction
    return jsonify(output)


if __name__ == '__main__':
    app.run(port=5000, debug=True)