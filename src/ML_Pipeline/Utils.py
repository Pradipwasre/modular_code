# Utils.py
import os
from keras.models import load_model, save_model as keras_save_model

def save_model(model, columns, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the model in the Keras format
    model.save(os.path.join(output_dir, "deep-ae-model.keras"))  # Using .keras extension

def load_model(model_path):
    return keras_load_model(model_path)  # Loading the model
