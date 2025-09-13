import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

print("Starting Neural Network training process...")

# --- 1. Load the "Answer Key" Dataset ---
# This is the dataset created by our first, rule-based model.
try:
    df = pd.read_csv('data/products_with_scores.csv')
    # Handle any potential missing reviews
    df['reviews'] = df['reviews'].fillna('')
    print(f"Successfully loaded dataset with {len(df)} products.")
except FileNotFoundError:
    print("Error: 'data/products_with_scores.csv' not found.")
    print("Please run the initial 'model_trainer.py' script first to generate this file.")
    exit()

# --- 2. Prepare the Data for the Neural Network ---

# The 'X' is our input (the feature): the review text.
# The 'y' is our output (the labels): the four scores we want to predict.
X = df['reviews']
y = df[['environmental_impact_score', 'labor_rights_score', 'animal_welfare_score', 'corporate_governance_score']].values

# A neural network only understands numbers, not text.
# We use a TF-IDF Vectorizer to convert the text into a numerical matrix.
# It identifies the most important words across all reviews.
print("Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# We save this vectorizer so we can use the exact same one for predictions later.
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("Saved TF-IDF vectorizer to 'tfidf_vectorizer.pkl'.")


# --- 3. Split Data into Training and Testing Sets ---
# We train the model on the training set and then test its performance
# on the unseen testing set to make sure it's learning correctly.
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
print(f"Data split into {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples.")

# --- 4. Build the Neural Network Architecture ---
print("Building the neural network model...")

model = tf.keras.Sequential([
    # Input layer: The shape must match the number of features from our TF-IDF vectorizer.
    tf.keras.layers.Input(shape=(X_train.shape[1],), sparse=True),
    # Hidden layers: These layers learn complex patterns in the data.
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2), # Dropout helps prevent overfitting
    tf.keras.layers.Dense(64, activation='relu'),
    # Output layer: It has 4 neurons, one for each ethical score we are predicting.
    # 'linear' activation is used for regression tasks (predicting a continuous number).
    tf.keras.layers.Dense(4, activation='linear')
])

# --- 5. Compile the Model ---
# We define the optimizer, the loss function, and any metrics to track.
model.compile(optimizer='adam',
              loss='mean_squared_error', # A common loss function for regression
              metrics=['mean_absolute_error'])

model.summary()

# --- 6. Train the Model ---
print("\nStarting model training...")
# An epoch is one full pass through the entire training dataset.
# We'll train for 20 epochs.
history = model.fit(X_train.toarray(), y_train,
                    epochs=20,
                    batch_size=32,
                    validation_data=(X_test.toarray(), y_test),
                    verbose=1)

print("\nTraining complete!")

# --- 7. Evaluate and Save the Model ---
loss, mae = model.evaluate(X_test.toarray(), y_test, verbose=0)
print(f"\nModel performance on test data:")
print(f"  - Mean Squared Error: {loss:.4f}")
print(f"  - Mean Absolute Error: {mae:.4f}")

# Save the trained model to a file. This is our final "brain".
model.save('ethical_model.keras')
print("\nSuccessfully saved the trained neural network to 'ethical_model.keras'.")
print("You are now ready to update the backend server to use this model!")
