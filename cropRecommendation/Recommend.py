# import joblib
# import numpy as np

# model = joblib.load("RandomForest.pkl")
# print()

# input_data = [[float(input()) for i in range(7)]]


# input_array = np.array(input_data)
# predictions = model.predict(input_array)

# print("Recommendation of crop:", predictions)

import joblib
import numpy as np
import gradio as gr
model = joblib.load("RandomForest.pkl")
print()

def crop_prediction(*inputs):
    input_data = np.array([inputs])
    predictions = model.predict(input_data)
    return predictions[0]

inputs = [gr.inputs.Number(label=f"Input {i+1}") for i in range(7)]

gr.Interface(fn=crop_prediction, inputs=inputs,outputs="text",title="Crop Recommendation", description="Enter seven vital values to get a recommendation of the crop.").launch()