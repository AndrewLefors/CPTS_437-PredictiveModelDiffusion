#Import Libraries and load data for training and testing the model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
import os
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = '/content/drive/My Drive/Diffusion_pureSolvent_cdata.csv'
data = pd.read_csv(file_path)
data = data.iloc[:, 1:]  # Drop the first column if it's an index
data = data.drop(data.columns[-2], axis=1)  # Drop the second to last column

# Splitting the data
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
#X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#Define the model to build
#This model uses HyperpParameter choice to get the most optimal number of neurons per layer, number of layers, and activation functions
def build_model(hp):
    model = Sequential()

    # First layer
    model.add(Dense(
        units=148,
        activation=hp.Choice('activation_input', ['relu', 'tanh']),
        input_shape=(X_train.shape[1],)
    ))

    # Dynamic number of hidden layers
    for i in range(hp.Int('n_layers', 1, 5)):
        model.add(Dense(
            units=hp.Int('units_' + str(i), min_value=1, max_value=20, step=2),
            activation=hp.Choice('activation_' + str(i), ['relu', 'tanh'])
        ))
        model.add(Dropout(hp.Float('dropout_' + str(i), min_value=0.1, max_value=0.5, step=0.05)))

    # Output layer
    model.add(Dense(1, activation='linear'))

    # Compile model
    model.compile(
        optimizer=Adam(hp.Float('learning_rate', min_value=1e-7, max_value=1e-2, sampling='LOG')),
        loss='mean_squared_error'
    )
    return model

#Project Directory and File name for Hyperparameter tuning file. 
#For this project, models HPChoice_delta_v3 and v4 were used
unique_dir = f"drive/MyDrive/CPTS_437_Tuner"
unique_project_name = f"hyperparam_tuning_final_HPChoice_delta_v3"

# Hyperparameter tuning
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=7,
    executions_per_trial=3,
    directory=unique_dir,
    project_name=unique_project_name
)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=150, restore_best_weights=True)

tuner.search(
    X_train, y_train,
    epochs=1000,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping]
)

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Train the best model with more epochs and early stopping
history = best_model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=32,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor='loss', patience=250, restore_best_weights=True)]
)

# Make predictions and evaluate % adherence
predictions = best_model.predict(X_test)
percentage_differences = 100 * abs(predictions.flatten() - y_test) / y_test

# Calculate how many predictions are within 5% of the actual values
within_5_percent = np.mean(percentage_differences <= 25)
print(f"Percentage of Predictions Within 25% of Actual Values: {within_5_percent * 100}%")

# Calculate percentage adherence

percentage_adherence = np.where(y_test != 0, percentage_differences, 0)

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(percentage_adherence, bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Percentage Adherence of Predictions')
plt.xlabel('Percentage Adherence')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Test metrics
test_mse = mean_squared_error(y_test, predictions)
test_mae = mean_absolute_error(y_test, predictions)
print(f"Test MSE: {test_mse}, Test MAE: {test_mae}")

# Reorder the data
# Sorting the actual values and predictions
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr

# Plotting
plt.figure(figsize=(12, 6))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.ylim(0, 1e-6)  # Set the y-axis maximum to 1e-4


# Convert y_test and predictions to 1-dimensional arrays if they're not already
y_test_1d = np.ravel(y_test)
predictions_1d = np.ravel(predictions)

# Calculate the Pearson correlation coefficient
r_value, _ = pearsonr(y_test_1d, predictions_1d)

# Plotting with Seaborn
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test_1d, y=predictions_1d, alpha=0.5)

# Adding a line for perfect predictions
plt.plot(y_test_1d, y_test_1d, color='red', label='Perfect Prediction Line')

# Setting plot title and labels
plt.title(f'Actual vs Predicted Diffusion Values\nPearson R: {r_value:.2f}')
plt.xlabel('Actual Diffusion')
plt.ylabel('Predicted Diffusion')
plt.legend()
plt.axis('equal')
plt.show()