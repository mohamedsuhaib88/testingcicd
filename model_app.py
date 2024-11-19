#import necessary libraries

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()


class Input(BaseModel):
    department: object
    region: object
    education: object
    gender: object
    recruitment_channel: object
    no_of_trainings: float
    age: float
    previous_year_rating: float
    length_of_service: float
    KPIs_met_80_percent: float
    awards_won: float
    avg_training_score: float

class Output(BaseModel):
    IsPromoted: int

@app.post("/predict")

def predict(data: Input) -> Output:
    # input
    # dataframe thru list
    X_input = pd.DataFrame([[data.department, data.region, data.education, data.gender, data.recruitment_channel, data.no_of_trainings, data.age, data.previous_year_rating, data.length_of_service, data.KPIs_met_80_percent, data.awards_won, data.avg_training_score]])
    X_input.columns = ['department', 'region', 'education', 'gender', 'recruitment_channel', 'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'KPIs_met_80_percent', 'awards_won', 'avg_training_score']

    # dataframe thru dictionary (valid)
    #X_input = pd.DataFrame([{'CONSOLE':  data.CONSOLE,'YEAR':  data.YEAR,'CATEGORY':  data.CATEGORY,'PUBLISHER':  data.PUBLISHER,'RATING':  data.RATING,'CRITICS_POINTS':  data.CRITICS_POINTS,'USER_POINTS':  data.USER_POINTS}])
   
    print(X_input)
    # load the model
    model = joblib.load('wns_pipeline_model.pkl')

    #predict using the model
    prediction = model.predict(X_input)

    # output
    return Output(IsPromoted = prediction)
