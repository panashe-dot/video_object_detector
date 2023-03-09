%%writefile app.py
import streamlit as st
import tempfile
import cv2    
import math  
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16
from PIL import Image

st.title(" ASSIGNMENT 2")
st.write("OBJECT DETECTION IN VIDEOS")
def uploadFile():
  f = st.file_uploader("Upload a video...", type=["mp4", "mov","avi"])

  tempVideo = tempfile.NamedTemporaryFile(delete=False) 

  if f is not None: 
    tempVideo.write(f.read())
  return tempVideo.name
        
def splitVideo(videoPath):
  """Split video uploaded into frames"""
  count = 0
  cap = cv2.VideoCapture(videoPath)   # capturing the video from path

  frameRate = cap.get(5) 

  tempImage = tempfile.NamedTemporaryFile(delete=False) 

  x=1

  while(cap.isOpened()):
    frameId = cap.get(1) 
    ret, frame = cap.read()
    if (ret != True):
      break
    if (frameId % math.floor(frameRate) == 0):
      #storing the frames 
      tempImage = videoPath.split('.')[0] +"_frame%d.jpg" % count;count+=1
      cv2.imwrite(tempImage, frame)
      frames.append(tempImage)
  cap.release() 
  return frames,count

def classifyObjects():  
  """Classify objects in frames using InceptionV3 and return array with objects detected"""
  from tensorflow.keras.applications.vgg16 import VGG16
  model =VGG16()
    
  classifications = [] #array to save classifications

  frames,count = splitVideo(videoFile)

  for i in range(count):    
    image = load_img(frames[i], target_size=(224, 224)) 
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)   
    prediction = model.predict(image)  # predict the probability 
    label = decode_predictions(prediction)    
    label = label[0][0] # retrieve the most likely result
    result =  label[1]
    classifications.append(result)
  return classifications
def searchInFrames(object_):
  """Search for object queried by user in frames from video"""
  indexes = []
  classifications = classifyObjects()
  if object_ in classifications:
    for i in range(len(classifications)):
      if classifications[i] == object_:
        index = classifications.index(object_)
        indexes.append(index)
        filePath = frames[index]
        img = load_img(filePath, target_size = (224, 224, 224))
        detected_paths.append(filePath)
    for i in range(len(indexes)):
      st.image(frames[i], width=224)
  else:
    st.write("Object not found!")

videoFile = uploadFile()
user_input = st.text_input("Enter object name: ")

if st.button('Classify'):  
  frames =[]
  detected_paths = []
  searchInFrames(user_input)
  st.write("")