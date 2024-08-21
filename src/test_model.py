import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json

def test_model():
    # Load dataset
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    # Load the model
    model = joblib.load('model.pkl')

    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    
    print(f"Test accuracy: {accuracy}")
    print(f"Classification Report: {json.dumps(report, indent=2)}")

    # Assert accuracy is above threshold
    assert accuracy > 0.9, "Model accuracy is below 90%!"

if __name__ == "__main__":
    test_model()
