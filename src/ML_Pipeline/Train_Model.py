# Train_Model.py
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from ML_Pipeline.Utils import save_model  # Ensure Utils.py has the save_model function

def train_model(data_path, output_dir):
    print("Loading data...")
    data = pd.read_csv(data_path)  # Load your actual data
    print("Data loaded into pandas dataframe")
    
    # Print the columns to check for the target name
    print("Available columns in the dataset:", data.columns.tolist())

    # Preprocessing steps (example)
    print("Preprocessing started....")
    # Your preprocessing logic goes here (e.g., normalization, encoding, etc.)
    print("Preprocessing completed....")

    # Define and train your model
    ml_model = Sequential()
    ml_model.add(Dense(64, activation='relu', input_shape=(data.shape[1]-1,)))  # Use number of features as input shape
    ml_model.add(Dense(32, activation='relu'))
    ml_model.add(Dense(1, activation='sigmoid'))  # Adjust this based on your actual output

    ml_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Update the target column name to match your data
    target_column_name = 'Class'  # Change this to the actual name of your target column

    # Verify if the target column exists
    if target_column_name not in data.columns:
        raise KeyError(f"'{target_column_name}' column not found in the dataset.")
    
    target = data[target_column_name]  # Use the actual target column name
    features = data.drop(target_column_name, axis=1)  # Adjust to drop the target column
    ml_model.fit(features, target, epochs=10, batch_size=32)  # Adjust epochs and batch size as needed

    # Save the trained model
    save_model(ml_model, columns=data.columns.tolist(), output_dir=output_dir)
    print(f"Model saved in: {output_dir}")
