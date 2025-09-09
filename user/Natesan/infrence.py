
import pickle
import pandas as pd



df = pd.DataFrame(
    [[3.28,2.0,'Medium','Clear',2.88,1.76,78.04]],
    columns=[
        'Trip_Distance_km', 
        'Passenger_Count',
        'Traffic_Conditions',
        'Weather',
        'Base_Fare'
        ,'Per_Km_Rate'
        ,'Trip_Duration_Minutes'
    ]
)

with open(r"F:\Data science\user\Natesan\ml model.pkl", "rb") as f:
    pipeline = pickle.load(f)

print(pipeline.predict(df))
