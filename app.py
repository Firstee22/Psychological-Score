import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic  import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware #to connect the backend and frontend with html,css,js script

model= joblib.load('mental_health_model.pkl')

app=FastAPI()

#this is the code for CORSMiddleware
app.add_middleware(        
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# A first Pydantic Model
class StudentData(BaseModel):
    Age                     :int =Field(...,ge=10,le=100) #... means the important, ge means greater than or equal to, le means less than or equal to
    Gender                  :Literal['Male','Female'] #specify that only 2 options we have
    Country                 :str
    Academic_Level          :Literal['Undergraduate', 'Graduate', 'High School']
    Most_Used_Platform      :Literal[ 'Facebook', 'LinkedIn', 'Instagram', 'Snapchat', 'Twitter', 
                                     'YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte', 
                                     'WhatsApp', 'WeChat']
    Purpose_Of_Use          :Literal['Networking', 'Education', 'Entertainment', 'News']
    Avg_Daily_Usage_Hours   :float =Field(...,ge=0,le=24) #the number should not and can not be negative and must be less than 24 hours
    Daily_Unlocks           :int  = Field(...,ge=0)
    Study_Hours             :float =Field(...,ge=0,le=24)
    Physical_Activity_Hours :float =Field(...,ge=0,le=24)
    Sleep_Hours_Per_Night   :float =Field(...,ge=0,le=24)
    Stress_Level            :Literal['Medium', 'Low', 'Very High', 'High']


# Describe what we send back
class PredictionResponse(BaseModel): #this just checks the value of prediciton to be float or not
    predicted_mental_health_score: float 


@app.get("/")
def greet():
    return {"Hello, Welcome to Mental Health Detection API"}


top_countries=['India','USA','Canada','Australia','UK','Germany','Mexico','Turkey','France','Spain']
@app.post("/predict",response_model=PredictionResponse)
def predict(data:StudentData):  #data is object and StudentData is class
    group_country=data.Country if data.Country in top_countries else 'Other'

    #input_row is the df the model will take as input to predict the output
    #this df values will be provided by the user through the API
    input_row=pd.DataFrame([{
        'Age'                           :data.Age,
        'Gender'                        :data.Gender,
        'Country'                       :data.Country,
        'Academic_Level'                :data.Academic_Level,
        'Most_Used_Platform'            :data.Most_Used_Platform,
        'Purpose_Of_Use'                :data.Purpose_Of_Use,
        'Avg_Daily_Usage_Hours'         :data.Avg_Daily_Usage_Hours,
        'Daily_Unlocks'                 :data.Daily_Unlocks,
        'Study_Hours'                   :data.Study_Hours,
        'Physical_Activity_Hours'       :data.Physical_Activity_Hours,
        'Sleep_Hours_Per_Night'         :data.Sleep_Hours_Per_Night,
        'Stress_Level'                  :data.Stress_Level,
        'Grouped_Country'               :group_country  #because we did feature engineering
    }])
    prediction=model.predict(input_row)[0]  #model will predict the output based on the input_row values
    return PredictionResponse(predicted_mental_health_score=round(prediction,2))



