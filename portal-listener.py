#!/bin/python2
import cv2
import serial
import numpy
import math

####WAT BETA TESTING
#dummyz = 50 #For testing



#################################
#	Initial Variables	###############################################
#################################

########################
# BEHAVIORAL VARIABLES #
########################

stillalive = 1 #Is our robot still alive?
robotstatus = 1 # 1=active, 2=sleeping, 3=freakout
sleepcoordinates=(0,0) #Sleep coordinate tuples

###################
# AUDIO VARIABLES #
###################

#Location/Name of sound
#activesound = 
#sleepsound = 
#freaksound = 

# Variables for keeping track of what's been played (when implemented)
#activesoundplayed = 0
#sleepsoundplayed = 0
#freaksoundplayed = 0


#####################
# SERIAL COMPONENTS #
#####################

#defines the port for the serial, baud 9600.
#Note: The '/dev/tty' bit may differ between distributions. You'll have to figure out how your distro mounts the arduino.
#shootybrain = serial.Serial('/dev/tty.usbserial',9600)

#############################
# Video setup and filtering #
#############################

#initialize the video capture input
capture = cv2.VideoCapture(0)

#Check for making sure the capture stream is working
if capture.isOpened(): # try to get the first frame
	stillalive, frame = capture.read()
else:
	stillalive = False

#Values to downsize the video to
downsizeheight = 120
downsizewidth = 160

#Initial settings for coordinates
rowcoordinate=1
colcoordinate=1

#Color of Circle
circlecolor = numpy.array([255,0,0])

#Define color to target
#Note: This uses a hue ranging from 0 to 179, not 0 to 360.
#Uses HSV values
####Bright Pink:
#lowercolor = numpy.array([155,100,75])
#uppercolor = numpy.array([175,255,255])
####Default Green (Folder Green):
lowercolor = numpy.array([40,100,75])
uppercolor = numpy.array([85,255,255])

#############
# Filtering #
#############

# 'kernel' defines the size of the filter's effect (if there even is a filter)
#should be uncommented only in case of filter
#kernel = numpy.ones((7,7),numpy.uint8)

##################
# Tracking Setup #
##################

##### MEANSHIFT/CAMSHIFT#####
# Should only be uncommented in case of camshift or meanshift

# Setup the termination criteria for the meanshift, either 3 iteration or move by at least 1 pt
#term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 3, 1 )

# set up the Region of Interest (ROI) for tracking
# setup initial location of window
#r,h,c,w = 350,50,300,50  # hardcoded the values, effectively initial position x,y and width, height.
#track_window = (c,r,w,h)
#roi = frame[r:r+h, c:c+w]
#hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

#The following lines were from an example, not sure if I'll need them later
#mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
#roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
#cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

##### NUMPY APPROACH VALUES #####
#Number of pixels to warrent an "object" being there
pixelthreshold = 50

#########################
#	Main Loop	#########################################################
#########################

while (stillalive == 1): #While there is still cake...
	
	####################
	# Video Processing #
	####################

	#Pull a new image
	stillalive, frame = capture.read()
	frame = cv2.resize(frame, (downsizewidth,downsizeheight))
	#Conversion of image to HSV
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	# threshold image of the hsv image, using upper and lower colors defined initially
	mask = cv2.inRange(hsv, lowercolor, uppercolor)

	#############
	# Filtering #
	#############	
	
	# Opening/Closing Filtering (http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)
	# Takes decent processing power, disable to improve speed
	# opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	# closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	
	# Combine mask with original picture.

	#Combine the mask and the original picture to include the original colors
	res = cv2.bitwise_and(frame,frame, mask=mask) #This is for calculating with the initial mask
	#res = cv2.bitwise_and(frame,frame, mask=closing) #This is for calculating with mask after closing
	#res = cv2.bitwise_and(frame,frame, mask=opening) #This is for calculating with mask after opening
	
	########################
	# Tracking Calculation #
	########################

	##################
	# NUMPY APPROACH #
	##################

	#convert mask to ones and zeros
	maskbin = numpy.divide(mask,255)
	rowsize,colsize,channels = frame.shape
	rownumbered = numpy.arange(rowsize)
	colnumbered = numpy.arange(colsize)
	
	#Sums calculated from mask, sums in the rows, columns, and overall
	allsum = numpy.sum(maskbin, axis=None)
	if (allsum>pixelthreshold):
		colsum = numpy.sum(maskbin, axis=0)
		rowsum = numpy.sum(maskbin, axis=1)

		#Multiply the rownumbered and the rowsum, then divide by the fullsum to get the correct position
	
		rowmultiplied = numpy.multiply(rowsum,rownumbered)
		colmultiplied = numpy.multiply(colsum,colnumbered)
		rowcoordinate = int(math.floor(numpy.sum(rowmultiplied) / allsum))
		colcoordinate = int(math.floor(numpy.sum(colmultiplied) / allsum))


		#Also, reset the sleep counter and make sure the status is "awake"
	else:   #If the max pixels are not met
		(rowcoordinate,colcoordinate)=sleepcoordinates #advance the sleep counter


	######################
	# MEANSHIFT APPROACH #
	######################
	
	#ret, track_window = cv2.meanShift(mask, track_window, term_crit) # Use meanshift command to calculate the position of the box
        #x,y,w,h = track_window # Divide the track_window variable into separate variables
	# print track_window # For debugging purposes only
        #img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2) #Draw the rectangle on the given imshow (first variable, should be 'frame' for the initial image.
        #cv2.imshow('img2',img2) #Unnecessary?
	
	#####################
	# CAMSHIFT APPROACH #
	#####################
	# Until I figure out a replacement for boxPoints(), I can't use this.
	#ret, track_window = cv2.CamShift(mask, track_window, term_crit)
	#pts = cv2.boxPoints(ret)
        #pts = np.int0(pts)
        #img2 = cv2.polylines(frame,[pts],True, 255,2)
        # cv2.imshow('img2',img2) #Unnecessary?

	###########
	# RESULTS #
	###########
	
	#Quick calculation of the coordinates to send to the arduino
	#This is needed only for Meanshift or camshift approach!
	#xcoor = x + (w/2)
	#ycoor = y + (h/2)
	#print '%d %d' % (xcoor,ycoor)
	
	########################
	# DISPLAYING THE VIDEO #
	########################

	#cv2.imshow('masked',mask) #Display Mask, for debug
	#cv2.imshow('multi',res) #Display initial-mask combo, for debug and coolness
	#cv2.imshow('frame',frame) #Regular image
	#print rowcoordinate
	#print colcoordinate
	print allsum
	coordinates = numpy.array([rowcoordinate,colcoordinate])
	cv2.circle(frame,(colcoordinate,rowcoordinate),5,circlecolor,thickness=3,lineType=8) #draw a blue circle at the coordinates calculated, for the numpy approach
	cv2.circle(res,(colcoordinate,rowcoordinate),5,circlecolor,thickness=3,lineType=8) #draw a blue circle at the coordinates calculated, for the numpy approach
	
	#cv2.imshow('frame',img2)
	if (allsum>pixelthreshold):
		cv2.imshow('frame',frame)
		cv2.imshow('res',res)


	##############
	# BEHAVIORAL #
	##############

	if (sleepcounter>sleepcountermax):
		robotstatus = 2
	if (freakout=1):
		robotstatus = 3
	if (freakout=0):
		robotstatus = 2
		sleepsoundplayed = 1
		sleepcounter = 0
	


	##########
	# SERIAL #
	##########
		
	# read any input serial information
	#recentcommand = shootybrain.readline()
	
	# Send state
	#shootybrain.write(robotstatus)
	#if (status=1):
		#shootybrain.write(xcoordinate)
		#shootybrain.write(ycoordinate)
	
	
	#########
	# AUDIO #
	#########

	if (robotstatus==1 & activesoundplayed==0):
		#play attack sound
		activesound=1 #set sound as already played
	if (robotstatus==2 & sleepsoundplayed==0):
		#play sleep sound
		sleepsoundplayed=1
	if (robotstatus==3 & freaksoundplayed==0):
		# play freak sound
		freaksoundplayed=1

	##################
	# Shutdown Check #
	##################
	#Hit the escape key to quit
	key = cv2.waitKey(30) & 0xff
    	if key == 27:
        	break

cv2.destroyAllWindows()

