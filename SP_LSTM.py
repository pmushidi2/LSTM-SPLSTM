import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from splstm_2 import SP_LSTM_Cell, SP_LSTM
import tensorflow as tf
import time

# Import the SP_LSTM layer from the earlier code (you can add the layer to a separate .py file and import from there)
# from SP_LSTM import SP_LSTM

# Load the dataset
data = pd.read_csv("Dataset.csv")

# Preprocess the dataset
X = data[["Link_ID", "Time", "Moving_Average_throughput", "Instantaneous_throughput", "Time_average_throughput"]].values
y = data["Moving_Average_throughput"].values

# Calculate the average throughput for each link
average_throughput = data.groupby("Link_ID")["Moving_Average_throughput"].mean().sort_values(ascending=False)

# Normalize the input features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Split the data into training and testing sets
train_size = int(0.8 * len(X_normalized))
X_train, X_test = X_normalized[:train_size], X_normalized[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape the input features for LSTM
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Step-based decay schedule
initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

# Passing the schedule to the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=0.5)

# Design the model architecture
model = Sequential()

# Add the first SP_LSTM layer to the model
model.add(SP_LSTM(units=64, input_dim=X_train.shape[2], return_sequences=True))

# Add the second SP_LSTM layer
model.add(SP_LSTM(units=64, input_dim=64, return_sequences=False)) # input_dim is equal to the units of the previous layer

# Add a dense output layer
model.add(Dense(units=1))

# Compile the model
model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=['mae'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Start timer for training
start_time = time.time()

# Train the model
# model.fit(X_train, y_train, epochs=10, callbacks=[callback], batch_size=32)
model.fit(X_train, y_train, epochs=10, batch_size=32, callbacks=[callback], validation_data=(X_test, y_test))

# Print training duration
print("Training duration: {} seconds".format(time.time() - start_time))


# Start timer for prediction
start_time = time.time()

# Rest of the code remains the same...
# Calculate training accuracy
y_train_pred = model.predict(X_train)

# Print prediction duration
print("Prediction duration: {} seconds".format(time.time() - start_time))

train_mse = np.mean((y_train_pred[:500]  - y_train[:500]) ** 2)
train_accuracy = 1 - (train_mse / np.var(y_train))
print("Training Accuracy:", train_accuracy)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model performance
mse = np.mean((predictions[:500] - y_test[:500]) ** 2)
print("Mean Squared Error:", mse)

# Calculate prediction accuracy
accuracy = 1 - (mse / np.var(y_test))
print("Prediction Accuracy:", accuracy)

# Save the predicted output to a CSV file
predicted_output = pd.DataFrame()
predicted_output["Link_ID"] = data["Link_ID"].values[train_size:]
predicted_output["Time"] = data["Time"].values[train_size:]
predicted_output["Current_Moving_Average_throughput"] = y_test
predicted_output["Predicted_Moving_Average_throughput"] = predictions

# Save the predicted output to a CSV file
predicted_output.to_csv("predicted_output.csv", index=False)
print("Predicted output saved to predicted_output.csv")

unique_links = predicted_output["Link_ID"].unique()
for link in unique_links:
    link_data = predicted_output[predicted_output["Link_ID"] == link]
    link_data.to_csv(f"predicted_output_link_{link}.csv", index=False)
    print(f"Predicted output for Link {link} saved to predicted_output_link_{link}.csv")

# Print links from most congested to least congested before prediction
print("Links sorted by Moving_Average_throughput (before prediction):")
for link_id in average_throughput.index:
    print(link_id)

# Plot the graph comparing current Moving_Average_throughput with Predicted_Moving_Average_throughput for each link
for link in unique_links:
    link_data = predicted_output[predicted_output["Link_ID"] == link]
    plt.figure(figsize=(10, 6))
    plt.plot(link_data["Time"].values, link_data["Current_Moving_Average_throughput"].values,
             label="Current Moving_Average_throughput")

    #plt.plot(link_data["Time"], link_data["Current_Moving_Average_throughput"], label="Current Moving_Average_throughput")
    # plt.plot(link_data["Time"], link_data["Predicted_Moving_Average_throughput"], label="Predicted Moving_Average_throughput")
    plt.plot(link_data["Time"].values, link_data["Predicted_Moving_Average_throughput"].values,
             label="Predicted Moving_Average_throughput")

    plt.xlabel("Time")
    plt.ylabel("Throughput (Mbps)")
    plt.title(f"Comparison of Current and Predicted Moving Average Throughput - Link {link}")
    plt.legend()
    plt.show()

# Sort links by Moving_Average_throughput (after prediction)
predicted_output["Moving_Average_throughput"] = average_throughput[predicted_output["Link_ID"]].values
sorted_links = predicted_output.groupby("Link_ID")["Moving_Average_throughput"].mean().sort_values(ascending=False)

# Print links from most congested to least congested after prediction
print("Links sorted by Moving_Average_throughput (after prediction):")
for link_id in sorted_links.index:
    print(link_id)



