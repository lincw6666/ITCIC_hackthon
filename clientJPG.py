import cv2
import numpy as np
import socket
import sys
import pickle
from PIL import Image # please pip3 install Pillow
import struct
import time
from io import BytesIO
import subprocess

def NumpyToJPG(arr):
	ret, buf = cv2.imencode( '.jpg', arr)
	out = buf.tostring()
	return out

def NumpyToPickle(arr):
	return pickle.dumps(arr)

# Open 2 camera.
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)

# Open socket.
clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('192.168.43.137',8000))

count = 0
while True:
	count += 1

	# Get a frame in the captured video.
	ret, frame1 = cap1.read()
	ret, frame2 = cap2.read()
	
	data = NumpyToJPG(frame1)
	if count == 10:
		clientsocket.sendall(struct.pack("L", len(data))+data)
		detect = clientsocket.recv(4096).decode().split(',')
		if int(detect[0]) > 0:
			process = subprocess.Popen(["python", "googleSpeech.py", detect[0], detect[1]])
	
	data = NumpyToJPG(frame2)
	if count == 10:
		clientsocket.sendall(struct.pack("L", len(data))+data)
		detect = clientsocket.recv(4096).decode().split(',')
		if int(detect[0]) > 0:
			process = subprocess.Popen(["python", "googleSpeech.py", detect[0], detect[1]])
		count = 0

	print ("hi %d"%count)
