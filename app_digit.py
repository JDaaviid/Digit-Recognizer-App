# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:45:33 2021

@author: JoseDavid
"""
from tkinter import *
from PIL import ImageGrab, Image, ImageDraw
import PIL
import win32gui
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import PIL.ImageOps 


class App():
  
    def __init__(self):  
      self.root = Tk()
      self.root.resizable(False, False)
      self.root.title("Digit recognizer")
      self.root.geometry("800x470")
      
      self.blackboard = Canvas(self.root, width=400, height=400, bg="white", cursor="cross")
      
      self.clear_button = Button(self.root, text="Clear", width=10, command=self.clear)
      self.label = Label(self.root, text="Drawing...", width=30, font=("Verdana", 16))
      self.recognise_button = Button(self.root, text="Recognise", width=10, command=self.recognise)
      
      #Grid structure
      self.blackboard.grid(row=0, column=0, pady=10, padx=10)
      self.clear_button.grid(row=1, column=0)
      self.label.grid(row = 0, column=1, padx=0)
      self.recognise_button.grid(row=1, column=1)
      
      width= int(self.blackboard['width'])
      height= int(self.blackboard['height'])
      
      white = (255, 255, 255)
      self.image = PIL.Image.new("RGB", (width, height), white)
      self.draw = ImageDraw.Draw(self.image)
      self.blackboard.bind( "<B1-Motion>", self.paint ) #The mouse is moved, with mouse button 1 (leftmost button) being held down
    
    def paint(self, event):
      x1, y1 = ( event.x - 10 ), ( event.y - 10 )
      x2, y2 = ( event.x + 10 ), ( event.y + 10 )
      self.blackboard.create_oval(x1, y1, x2, y2, fill='black')
      self.draw.line([x1, y1, x2, y2],fill="black", width=20)
    
    def clear(self):
      self.blackboard.delete("all")
      self.draw.rectangle((0, 0, 400, 400), width=0, fill='white')
    
    def recognise(self):
      #Get the image
      self.image.save("image.jpg")
      
      #Apply the model
      im = self.image.resize((28,28))
      #Invert colors (the cnn is trained with these colored pictures)
      im = PIL.ImageOps.invert(im)
      #Convert RGB image to grayscale
      im = im.convert('L')
      plt.imshow(im, cmap="gray")
      im = np.array(im)
      #Adjust images to be valid for the input of the CNN (The CNN model will require one more dimension)
      im = im.reshape(1, 28,28,1)
      im = im.astype('float32')
      im /= 255
      prediction = model.predict([im])[0]
      accuracy = max(prediction)
      accuracy = accuracy * 100
      accuracy = "{:.1f}".format(accuracy)
      #print("Prediction: " + str(np.argmax(prediction)) +" Accuracy: " + str(accuracy)+"%")
      self.label.configure(text="Prediction: {} Accuracy: {}%".format(np.argmax(prediction), accuracy))
     
model = load_model("mnist_989%.h5")   
app = App()
    
app.root.mainloop()