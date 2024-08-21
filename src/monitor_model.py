import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests

def monitor_model():
    # Load dataset
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    # Load the model
    model = joblib.load('model.pkl')

    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Monitoring accuracy: {accuracy}")

    # Trigger retraining if accuracy drops below a threshold
    if accuracy < 0.9:
        print("Accuracy dropped below threshold! Triggering retraining...")
        # Example: You could send a request to trigger a retraining job or use GitHub API to create an issue
        # requests.post('YOUR_RETRAINING_TRIGGER_ENDPOINT')
        # or
        # raise an alert by creating an issue on GitHub
        # issue_title = "Model retraining required"
        # issue_body = f"Model accuracy dropped to {accuracy}, which is below the acceptable threshold."
        # requests.post(f"https://api.github.com/repos/USERNAME/REPO/issues", json={"title": issue_title, "body": issue_body})

if __name__ == "__main__":
    monitor_model()
