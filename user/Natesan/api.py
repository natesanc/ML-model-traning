
from flask import Flask, request # Imported Flask
import pandas as pd
import pickle

app = Flask(__name__) # Created a simple app

@app.get("/hi")
def hello():
    return {
        "message" : "Hello World",
        "status" : 200
    }

@app.post("/predict")
def model_predict():
    data = request.get_json()

    Trip_Distance_km = data.get("TripDistance")
    Passenger_Count = data.get("PassengerCount")
    Traffic_Conditions = data.get("TrafficConditions")
    Weather = data.get("Weather")
    Base_Fare = data.get("BaseFare")
    Per_Km_Rate = data.get("PerKmRate")
    Trip_Duration_Minutes = data.get("TripDurationMinutes")

    with open(r"F:\Data science\user\Natesan\ml model.pkl", "rb") as file:
        pipeline = pickle.load(file)

    df = pd.DataFrame(
    [[Trip_Distance_km,Passenger_Count,Traffic_Conditions,Weather,Base_Fare,Per_Km_Rate,Trip_Duration_Minutes]],
    columns=[
        'Trip_Distance_km', 
        'Passenger_Count',
        'Traffic_Conditions',
        'Weather',
        'Base_Fare',
        'Per_Km_Rate',
        'Trip_Duration_Minutes'
    ]
)
    
    prediction = pipeline.predict(df)

    return {
        "prediction" : prediction[0],
        "status" : 200
    }

if __name__ == "__main__":
    app.run(debug=True)
