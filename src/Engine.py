import os
from ML_Pipeline.Train_Model import train_model  # Importing the training function
from ML_Pipeline.Predict import predict_model  # Importing the prediction function

def main():
    print("Train - 0")
    print("Predict - 1")
    print("Deploy - 2")
    
    choice = int(input("Enter your value: "))

    if choice == 0:
        # Training
        data_path = input("Please enter the path to your training data: ")
        output_dir = input("Please enter the output directory for saving the model: ")
        print("Loading data...")
        # Call the train_model function
        train_model(data_path, output_dir)

    elif choice == 1:
        # Prediction
        model_path = input("Please enter the path to the model: ")
        input_data_path = input("Please enter the path to the input data for prediction: ")
        print("Loading model and making predictions on", input_data_path, "...")
        
        # Call the predict_model function with both required arguments
        predictions = predict_model(model_path, input_data_path)
        print("Predictions:", predictions)  # Display predictions

    elif choice == 2:
        # Deploy (You can implement your deploy logic here)
        print("Deployment functionality is not implemented yet.")

if __name__ == "__main__":
    main()
