import joblib

# Load the Orange3-trained model
orange_model = joblib.load("logistic_model.pkl")


sklearn_model = orange_model.skl_model 


joblib.dump(sklearn_model, "logistic_model_clean.pkl")

print("Standalone scikit-learn model saved as 'logistic_model_clean.pkl'.")