import os,sys
sys.path.append(os.getcwd())

from flask import Flask,render_template,jsonify,request
from src.logger import logging
from src.exception import CustomException
from src.pipelines.prediction_pipeline import CustomData,PredictionPipeline


app = Flask(__name__)

@app.route("/")
def home_page():
    return render_template("home.html")

@app.route("/predict",methods = ["POST",'GET'])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    else:
        input_data = CustomData(
            Airline = request.form.get('Airline'),
            Source = request.form.get('Source'),
            Destination = request.form.get('Destination'),
            Date_of_Journey = request.form.get('Date_of_Journey'),
            Total_Stops = request.form.get('Total_Stops'),
            Departure_Time = request.form.get('Departure_Time'),
        )
        input_data.preprocess_predict_data()
        final_input_data = input_data.get_data_as_dataframe()
        prediction_pipeline = PredictionPipeline()
        pred = prediction_pipeline.predict(final_input_data)

        result = round(pred[0],2)
        return render_template("form.html",final_result = result)


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
