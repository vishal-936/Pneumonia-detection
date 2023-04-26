from tensorflow.keras.models import load_model 
import numpy as np
import cv2

model = load_model('little_imporved_model.h5')

def predict_from_img(img):
  img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
  img = img/255.0
  img = np.expand_dims(img,axis = 0)
  output = model.predict(img)[0][0]
  return {'NORMAL':float(output),'PNEUMONIA':float(1-output)}

import gradio as gr
image = gr.inputs.Image(shape=(150,150))
label = gr.outputs.Label(num_top_classes=2)
gr.Interface(fn=predict_from_img, inputs=image, outputs=label,title = 'PNEUMONIA-DETECTION').launch()