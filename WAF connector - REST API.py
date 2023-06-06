#REST API code plust the dynamic classification of queries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '3' to show only errors

import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load LSTM model
model = load_model('lstm_model.h5')

@app.route('/checkMetadata', methods=['POST'])
def check_metadata():
    metadata = request.json
    
    # Log or print the metadata to the console
    print("Received JSON data: ", metadata) 
    
    path = metadata.get('path')

    # Add "/" to the start of path if it doesn't start with "/"
    if not path.startswith("/"):
        path = "/" + path

    # Prepare path for prediction
    path_encoded = tokenizer.texts_to_sequences([path])
    path_encoded = pad_sequences(path_encoded, maxlen=1202)

    # Predict
    prediction = model.predict(path_encoded)
    prediction = np.round(prediction).item()

    if prediction == 1:
        return jsonify({"result": "malign request"}), 200
    else:
        return jsonify({"result": "benign request"}), 200

if __name__ == "__main__":
    # 'cert.pem' and 'key.pem' are the paths to the certificate and private key files for encrypted traffic
    #Here we are implementing secure communication pattern
    app.run(host='0.0.0.0', port=443, ssl_context=('cert.pem', 'key.pem'))