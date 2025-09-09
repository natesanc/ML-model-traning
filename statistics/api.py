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

    Gender = data.get("Gender")
    Age = data.get("Age")
    Occupation = data.get("Occupation")
    City_Category = data.get("CityCategory")
    Stay_In_Current_City_Years = data.get("StayInCurrentCityYears")
    Marital_Status = data.get("MaritalStatus")
    Product_Category_1 = data.get("ProductCategory")

    with open(r"C:\Users\mhema\OneDrive\Desktop\DataScience\MachineLearning\regression.pkl", "rb") as file:
        pipeline = pickle.load(file)

    df = pd.DataFrame(
        [[Gender, Age, Occupation, City_Category, Stay_In_Current_City_Years, Marital_Status, Product_Category_1]],
        columns=[
            'Gender',
            'Age',
            'Occupation',
            'City_Category',
            'Stay_In_Current_City_Years',
            'Marital_Status',
            'Product_Category_1'
        ]
    )
    
    prediction = pipeline.predict(df)

    return {
        "prediction" : prediction[0],
        "status" : 200
    }

if __name__ == "__main__":
    app.run(debug=True)