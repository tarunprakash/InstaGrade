import tensorflow as tf
import numpy as np
import os
from __main__ import *
import cv2
from PIL import Image, ImageFilter, ImageEnhance
from random import randint
from MNIST import MNIST

class MNISTTester(MNIST):
    def __init__(self, model_path=None, data_path=None):
        MNIST.__init__(self, model_path, data_path)

        self.init()

    def init(self):

        self.init_session()

        self.load_model()

        if self.data_path is not None:
            self.load_training_data(self.data_path)

    def predict(self, filename):
        img = filename
        #img = self.loadImage(filename)

        number = self.sess.run(tf.argmax(self.model, 1), {self.X: img})[0]
        accuracy = self.sess.run(tf.nn.softmax(self.model), {self.X: img})[0]
        return {"number":number,"accuracy":accuracy[number]}
        

    def loadImage(self, filename):
        img = Image.open(filename).convert('L')
        contrast = ImageEnhance.Contrast(img)
        contrast = contrast.enhance(1000)
        contrast.save(filename)
        img = Image.open(filename).convert('L')
        # resize to 28x28
        img = img.resize((28, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        # normalization : 255 RGB -> 0, 1
        data = [(255 - x) * 1.0 / 255.0 for x in list(img.getdata())]

        # reshape -> [-1, 28, 28, 1]
        return np.reshape(data, (-1, 28, 28, 1)).tolist()

    def splitImage(self,filename,showImage=False):
        imgfile = Image.open(filename)
        im = cv2.imread(filename)
        numbers = []
        # Convert to grayscale and apply Gaussian filtering
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

        # Threshold the image
        ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the image
        image, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get rectangles contains each contour
        self.rects = [cv2.boundingRect(ctr) for ctr in ctrs]
        self.rects = [list(r) for r in self.rects]
        self.rects = [r for r in self.rects if (r[2]>0) and (r[3]>0) and (r[1]>0 and r[2]>0)]
        for i in range(len(self.rects)):
            self.rects[i][0] -= 3
            self.rects[i][1] -= 3
            self.rects[i][2] += 6
            self.rects[i][3] += 6
        """
        ## OLD SORTING
       for i in range(len(self.rects)):
            for j in range(i+1, len(self.rects)):
                if self.rects[i][1] in range(self.rects[j][1] - 200, self.rects[j][1] + 200) and self.rects[i][0] < self.rects [j][0]:
                    a = self.rects[i]
                    self.rects[i] = self.rects[j]
                    self.rects[j] = a
        self.rects.reverse()
        """
        self.rects1 =[]      
        ## REMOVING OPERATIONS AND EQUALS SIGN BASED ON HOW MUCH LOWER IT IS FROM THE REST OF THE NUMEBRS (60%)
        maxx = 0
        for i in self.rects:
          if i[3] > maxx:
           maxx = i[3]

        y_ratio = int(maxx * 0.7) ## MORE SELECTIVE = HIGHER PERCENT
        

        for i in range(len(self.rects)):
          if self.rects[i][3] > y_ratio:
            self.rects1.append(self.rects[i])
            
        """
        if self.rects1 != []:
            for item in self.rects1:
                del self.rects[self.rects1.index(item)]    
        """ 
        b = self.rects1
        self.rects1 = self.rects
        self.rects = b


        ## SORTING THE COORDS
        for i in range(len(self.rects)):
          for j in range(i+1, len(self.rects)):
            if self.rects[i][1] in range(self.rects[j][1] - y_ratio, self.rects[j][1] + y_ratio) and self.rects[i][0] < self.rects[j][0]:
                a = self.rects[i]
                self.rects[i] = self.rects[j]
                self.rects[j] = a

        self.rects.reverse()

        ## MAKING self.rects1 INTO THE REST
        for x in self.rects1:
          if x in self.rects:
            self.rects1.pop(self.rects.index(x))

        #print self.rects ## self.rects IS THE NUMBERS
        #print self.rects1 ## self.rects1 IS THE REST

        for rect in self.rects:
            # Draw the rectangles
            cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
            # Make the rectangular region around the digit
            leng = int(rect[3] * 1.6)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
            # Resize the image
            if roi.any():
                roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                roi = cv2.dilate(roi, (3, 3))

            cropRect = (rect[0],rect[1],rect[0]+rect[2],rect[1]+rect[3])
            imgCrop = imgfile.crop(cropRect)
            imgCrop.save("x.jpg")
            imgCrop1 = self.loadImage("x.jpg")
            numbers.append(imgCrop1)
        if showImage:
            cv2.imshow("Resulting Image with Rectangular ROIs", im)
            cv2.waitKey()
        return {"numbers":numbers,"rects":self.rects}

