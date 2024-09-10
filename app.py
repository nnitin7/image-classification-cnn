from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('cnn_model.h5')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Process input image
    # Add code to handle image input, pre-process, and predict using model
    prediction = np.argmax(model.predict(np.zeros((1, 32, 32, 3))))
    
    return render_template('index.html', prediction_text=f'Predicted Class: {prediction}')

if __name__ == '__main__':
    app.run(debug=True)
