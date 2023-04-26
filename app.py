from ForestCoverPrediction.pipeline.batch_prediction import start_batch_prediction
from ForestCoverPrediction.pipeline.training_pipeline import TrainPipeline
from ForestCoverPrediction.logger import logging
from ForestCoverPrediction.exception import ForestCoverPredictionException
from ForestCoverPrediction.entity.config_entity import TARGET_COLUMN
import os,sys

import streamlit as st
import pandas as pd

st.header("Forest Cover Type Prediction Application")

st.subheader("Please upload the input csv file:")
input_file = st.file_uploader("Upload the file")

print(__name__)

train_pipeline = TrainPipeline()
train_pipeline.run_pipeline()

transformer_object_path = train_pipeline.data_transformation_artifact.transformer_object_path
model_path = train_pipeline.model_train_eval_artifact.model_path
f1_train_score = train_pipeline.model_train_eval_artifact.f1_train_score
f1_test_score = train_pipeline.model_train_eval_artifact.f1_test_score


if __name__=="__main__":

     try:
        if input_file:
               logging.info("Starting the prediction")
               output_file_path = start_batch_prediction(input_file,transformer_object_path,model_path)
               df = pd.read_csv(output_file_path)
               prediction_column = df["Prediction"]
               dict_replace = {
                    1 : "Spruce/Fir"
                    2 : "Lodgepole Pine"
                    3 : "Ponderosa Pine"
                    4 : "Cottonwood/Willow"
                    5 : "Aspen"
                    6 : "Douglas-fir"
                    7 : "Krummholz"
                    }
               prediction_in_words = prediction_column.replace(dict_replace)

               output_df = pd.DataFrame({"prediction":prediction_column,
                                         "prediction_in_words": prediction_in_words})


               logging.info("Displaying the prediction result in webpage")
               st.subheader("Result")
               st.dataframe(output_df)
               st.subheader("F1 Score of the model")
               score_dict = {"Train Score" : f1_train_score,
                             "Test Score" : f1_test_score}

               st.dataframe(score_dict)

     except Exception as e:
          raise ForestCoverPredictionException(e,sys)






