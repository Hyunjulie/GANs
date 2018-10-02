import cv2
import numpy as np
import os

#Create a VideoCapture object and read from input file 
# If the input is from camera, pass 0 instead of video file name 
# Remember to put the video in the same folder as the code
cap = cv2.VideoCapture('zzanggu.mp4')
if(cap.isOpened()==False):
	print("Error opening video file")
try: 
	if not os.path.exists('data'):
		os.makedirs('data')
except OSError:
	print("Error: Creating directory of data")

currentFrame = 0 
filenum = 1

while(True):
	#Capture frame-by-frame
	ret, frame = cap.read()

	if currentFrame % 100 == 0: 
		#Saves image of the current frame in jpg file
		name = './data/frame' + str(filenum) + '.jpg'
		print("Creating...." + name)
		cv2.imwrite(name, frame)
		filenum += 1

	#To stop duplicate images 
	currentFrame += 1

#When done, release the capture 
cap.release()
cv2.destroyAllWindows()