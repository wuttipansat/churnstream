import joblib
import pandas as pd

from churnstream.core.config import get_settings

def main() -> str:
    settings = get_settings()

    model = joblib.load(settings.model_path)

    customer = pd.DataFrame(
        [
            {
                "CreditScore": 650,
                "Geography": "France",
                "Gender": "Male",
                "Age": 42,
                "Tenure": 2,
                "Balance": 0.0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 101348.88,
            }
        ]
    )

    prediction = int(model.predict(customer)[0])
    probability = float(model.predict_proba(customer)[0, 1])

    print(f"Prediction: {prediction}")
    print(f"Probability: {probability}")

if __name__ == "__main__":
    main()