import os
import sys

import tensorflow as tf
from tensorflow.keras.models import load_model

import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

import smtplib
from email.mime.text import MIMEText


def get_images(dir):
    img_size = 224
    images = [] 
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        img = Image.open(path)
        img = img.resize((img_size, img_size))
        images.append(img)

    images = np.array([np.array(img) for img in images])
    images = images / 255.0
    return images


image_directory = "/Users/jayasakthibalajig/Sakthi/Traffic/TrafficManagement/runs/Pics"
# image_directory = sys.argv[1]
images = get_images(image_directory)

model=load_model('model.h5', compile=False)
predictions = model.predict(images)
#model.summary()

# Select image to display
img_index = 0
# Get predicted class label
class_label = np.argmax(predictions[img_index])

# Display image and predicted class label
# plt.imshow(images[img_index])
# plt.axis('off')
# plt.title('Predicted class: ' + str(class_label))
# plt.show()

################################################
labels = ['accident', 'dense_traffic', 'fire', 'sparse_traffic']
count_fire=0
count_dense=0
count_sparse=0
count_accident=0

for i in range(len(images)):
    predicted_class = np.argmax(predictions[i])
    class_probability = predictions[i, predicted_class]
    print(f'Predicted class for {i+1}.jpg : {labels[predicted_class]}')
    if(labels[predicted_class]=="fire"):
        count_fire+=1
    elif labels[predicted_class] == "dense_traffic":
        count_dense+=1
    elif labels[predicted_class] == "accident":
        count_accident+=1
    elif labels[predicted_class] == "sparse_traffic":
        count_sparse+=1
    print('Class probability:', class_probability)

print()
print("Fire = ",count_fire)
print("Dense Traffic = ",count_dense)
print("Sparse Traffic = ",count_sparse)
print("Accident = ",count_accident)



if count_fire>200:
    print("")
    print("=========>Sending Mail<=============")
    sender_email = 'gjsbsakthi@gmail.com'
    sender_password = 'hdyc qqas icwq soge'
    recipient_email = 'ni3kfury@gmail.com'
    subject = 'Alert'
    body = 'Fire'
    print("Fire")
    
    message = MIMEText(body)
    message['Subject'] = subject
    message['From'] = sender_email
    message['To'] = recipient_email
    
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, message.as_string())
    print("------------>Email Sent<-----------")



elif count_dense>count_fire and count_dense>count_accident and count_dense>count_sparse:
    print("")
    print("=========>Sending Mail<=============")
    sender_email = 'gjsbsakthi@gmail.com'
    sender_password = 'hdyc qqas icwq soge'
    recipient_email = 'ni3kfury@gmail.com'
    subject = 'Alert'
    body = 'Dense Traffic'
    print("Dense")
    
    message = MIMEText(body)
    message['Subject'] = subject
    message['From'] = sender_email
    message['To'] = recipient_email
    
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, message.as_string())
    print("------------>Email Sent<-----------")




elif count_accident>count_fire and count_accident>count_dense and count_accident>count_sparse:
    print("")
    print("=========>Sending Mail<=============")
    sender_email = 'gjsbsakthi@gmail.com'
    sender_password = 'hdyc qqas icwq soge'
    recipient_email = 'ni3kfury@gmail.com'
    subject = 'Alert'
    body = 'Accident'
    print("Accident")
    
    message = MIMEText(body)
    message['Subject'] = subject
    message['From'] = sender_email
    message['To'] = recipient_email
    
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, message.as_string())

    print("------------>Email Sent<-----------")