from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the models
model_lstm = pickle.load(open('lstm_model.pkl', 'rb'))
model_gru = pickle.load(open('gru_model.pkl', 'rb'))

# Load the scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('Main.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    data = request.form

    # Convert the data to float and reshape for the model
    values = np.array([float(data[i]) for i in data]).reshape(1, -1)

    # Scale the values
    values_scaled = scaler.transform(values)

    # Reshape the data for LSTM and GRU
    values_reshaped = values_scaled.reshape(values_scaled.shape[0], 1, values_scaled.shape[1])

    # Make prediction using the models loaded from disk
    lstm_prediction = model_lstm.predict(values_reshaped)[0][0]
    gru_prediction = model_gru.predict(values_reshaped)[0][0]

    # Interpret the predictions
    lstm_result = "Potable Water" if lstm_prediction >= 0.5 else "Non-Potable Water"
    gru_result = "Potable Water" if gru_prediction >= 0.5 else "Non-Potable Water"

    # Return a response
    return render_template('Result.html', lstm_result=lstm_result, gru_result=gru_result)


if __name__ == '__main__':
    app.run(debug=True)
