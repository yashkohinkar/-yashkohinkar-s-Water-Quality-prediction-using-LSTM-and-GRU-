# Import necessary libraries
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras import Sequential

# Load and preprocess the dataset
# Assuming that the dataset is in a CSV file and the target variable is 'Potability'
df = pd.read_csv('water_potability.csv')
X = df.drop('Potability', axis=1)
Y = df['Potability']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
# Reshape the features for LSTM and GRU
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Build, compile, and train the LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model_lstm.add(Dense(1, activation='sigmoid'))  # Assuming binary classification
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_lstm.fit(X_train_reshaped, Y_train, epochs=50, batch_size=32, verbose=1)

# Build, compile, and train the GRU model
model_gru = Sequential()
model_gru.add(GRU(50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model_gru.add(Dense(1, activation='sigmoid'))  # Assuming binary classification
model_gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_gru.fit(X_train_reshaped, Y_train, epochs=50, batch_size=32, verbose=1)

# Evaluate the models
train_lstm_score = model_lstm.evaluate(X_train_reshaped, Y_train, verbose=0)
test_lstm_score = model_lstm.evaluate(X_test_reshaped, Y_test, verbose=0)
train_gru_score = model_gru.evaluate(X_train_reshaped, Y_train, verbose=0)
test_gru_score = model_gru.evaluate(X_test_reshaped, Y_test, verbose=0)

print('LSTM Model - Train Accuracy:', train_lstm_score[1], 'Test Accuracy:', test_lstm_score[1])
print('GRU Model - Train Accuracy:', train_gru_score[1], 'Test Accuracy:', test_gru_score[1])

# Save the models as pickle files
pickle.dump(model_lstm, open('lstm_model.pkl', 'wb'))
pickle.dump(model_gru, open('gru_model.pkl', 'wb'))
