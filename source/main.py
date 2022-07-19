import cv2
import os
import numpy as np

"""
region of interest function is used to mask parts of the image
so that the hough lines function can be implemented on the image
necessary for lane detection.
"""
def region_of_interest(img,vertices):
    mask = np.zeros_like(img) #creating a mask with the same image size
    cv2.fillPoly(mask,vertices,255) #Fill mask with 1 value joining the verticecs array 
    masked_image=cv2.bitwise_and(img,mask) #area of the image formed after excluding the area which is not needed.
    return masked_image
   
"""
This function creates a blank image on which the lines are drawn and
based on the value of the slope the color of the line is depicted in the video. 
"""
def draw_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
  # Loop through the lines array to use cv2.line function to depict each line on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2-y1)/(x2-x1)
            if slope>0:
                                           
                cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                cv2.line(blank_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
  # Blend the original image and the blank image
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

"""
The main function takes an image as an input and returns
the preprocessed image. 
"""
def preprocess(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #converts the image to grayscale
    value =100
    mat = np.ones(gray.shape,dtype = 'uint8')*value
    subtract = cv2.subtract(gray,mat) #making the gray image even more darker
    ret,thresh = cv2.threshold(subtract,130,145,cv2.THRESH_BINARY) #applying threshold to convert to binary image
    edge = cv2.Canny(thresh,100,200)  #canny edge detection
    height = image.shape[0]
    width = image.shape[1]
    roi_arr=[(width,height),(0,height),(width/2,height/2)]
    crop_img = region_of_interest(edge,np.array([roi_arr],np.int32))
    return crop_img

CWD_PATH = os.getcwd() # Get path to the current working directory
vid = cv2.VideoCapture('./input_videos/whiteline.mp4') 
result = cv2.VideoWriter('Results/simple_lane_detection.avi', cv2.VideoWriter_fourcc(*'MJPG'),25, (960,540))
while True:
    ret,frame = vid.read()
    if frame is None:
        break
    cropped= preprocess(frame)
    lines = cv2.HoughLinesP(cropped, rho=2, theta=np.pi/180,threshold=50, lines=np.array([]), minLineLength=10, maxLineGap=10)
    img = draw_lines(frame,lines)
    result.write(img)   
    cv2.imshow('Lane detection',img)
    if cv2.waitKey(1) & 0xFF == 27:
       break