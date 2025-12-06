from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application  

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 
  
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        try:
            gender = request.form.get('gender') or 'male'
            race_ethnicity = request.form.get('race_ethnicity') or 'group A'
            parental_level = request.form.get('parental_level_of_education') or "bachelor's degree"
            lunch = request.form.get('lunch') or 'standard'
            test_prep = request.form.get('test_preparation_course') or 'none'

            # Safely parse scores with sensible defaults
            try:
                reading_score = float(request.form.get('reading_score'))
            except Exception:
                reading_score = 50.0
            try:
                writing_score = float(request.form.get('writing_score'))
            except Exception:
                writing_score = 50.0

            data=CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level,
                lunch=lunch,
                test_preparation_course=test_prep,
                reading_score=reading_score,
                writing_score=writing_score
            )
            pred_df=data.get_data_as_data_frame()
            print(pred_df)
            print("Before Prediction")

            predict_pipeline=PredictPipeline()
            print("Mid Prediction")
            results=predict_pipeline.predict(pred_df)
            print("after Prediction")
            return render_template('home.html',results=results[0])
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print("Prediction error:", e)
            print(tb)
            return render_template('home.html', results=f"Error: {e}"), 500
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)        


