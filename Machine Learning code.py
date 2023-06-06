#Machine Learning code to train our WAF Component
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Load the data from a CSV file
data = pd.read_csv('merge_dataset.csv')

requests = data['Query'].values
labels = [1 if label == "malign" else 0 for label in data['Label'].values]

# Split the data into training and test sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(requests, labels, test_size=0.2, random_state=42)

# We use a tokenizer to prepare the requests for the LSTM
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X_train_raw)

X_train = tokenizer.texts_to_sequences(X_train_raw)
X_train = pad_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test_raw)
X_test = pad_sequences(X_test, maxlen=X_train.shape[1]) # make sure all sequences have the same length

print(X_train)
print(X_test)
# Create LSTM model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=32, input_length=X_train.shape[1]),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Convert labels to numpy array
y_train = np.array(y_train)
y_test = np.array(y_test)

# Train model
model.fit(X_train, y_train, epochs=4, batch_size=128, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Generate predictions on the test set
y_pred = model.predict(X_test)
y_pred = [1 if p > 0.5 else 0 for p in y_pred]

# Calculate metrics
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print('F1-score:', f1)
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)

# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the model
model.save('lstm_model.h5')