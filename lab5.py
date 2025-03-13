import pandas as pd
from collections import defaultdict

# Sample dataset based on the image
data = [
    ("Sunny", "Hot", "No"), ("Sunny", "Hot", "No"), ("Overcast", "Hot", "Yes"),
    ("Rainy", "Mild", "Yes"), ("Rainy", "Cool", "Yes"), ("Rainy", "Cool", "No"),
    ("Overcast", "Cool", "Yes"), ("Sunny", "Mild", "No"), ("Sunny", "Cool", "Yes"),
    ("Rainy", "Mild", "Yes"), ("Sunny", "Mild", "Yes"), ("Overcast", "Mild", "Yes"),
    ("Overcast", "Hot", "Yes"), ("Rainy", "Mild", "No")
]

df = pd.DataFrame(data, columns=["Weather", "Temperature", "Play"])

# Function to calculate probabilities
def naive_bayes_predict(weather, temperature):
    # Calculate prior probabilities
    total_yes = len(df[df["Play"] == "Yes"])
    total_no = len(df[df["Play"] == "No"])
    total = len(df)

    P_yes = total_yes / total
    P_no = total_no / total

    # Calculate likelihoods
    likelihoods = defaultdict(lambda: {"Yes": 0, "No": 0})

    for feature in ["Weather", "Temperature"]:
        for value in df[feature].unique():
            count_yes = len(df[(df[feature] == value) & (df["Play"] == "Yes")])
            count_no = len(df[(df[feature] == value) & (df["Play"] == "No")])
            likelihoods[value]["Yes"] = count_yes / total_yes if total_yes else 0
            likelihoods[value]["No"] = count_no / total_no if total_no else 0

    # Compute posterior probabilities
    P_play_yes = P_yes * likelihoods[weather]["Yes"] * likelihoods[temperature]["Yes"]
    P_play_no = P_no * likelihoods[weather]["No"] * likelihoods[temperature]["No"]

    # Normalize probabilities
    if P_play_yes + P_play_no > 0:
        P_play_yes /= (P_play_yes + P_play_no)
        P_play_no /= (P_play_yes + P_play_no)

    return "Yes" if P_play_yes > P_play_no else "No"

# Test cases
test_cases = [("Sunny", "Cool"), ("Rainy", "Hot"), ("Overcast", "Mild"), ("Sunny", "Mild"), ("Rainy", "Mild")]
expected_outputs = ["Yes", "No", "Yes", "Yes", "No"]

# Running predictions
for (weather, temperature), expected in zip(test_cases, expected_outputs):
    prediction = naive_bayes_predict(weather, temperature)
    print(f"Input: (Weather={weather}, Temperature={temperature}) → Predicted: {prediction}, Expected: {expected}")
