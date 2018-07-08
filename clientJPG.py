import cv2
import numpy as np
import socket
import sys
import pickle
from PIL import Image # please pip3 install Pillow
import struct
import time
from io import BytesIO

def NumpyToJPG(arr):
	ret, buf = cv2.imencode( '.jpg', arr)
	out = buf.tostring()
	return out

def NumpyToPickle(arr):
	return pickle.dumps(arr)

cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)
clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('192.168.43.137',8000))
count = 0
while True:
	count += 1
	ret, frame1 = cap1.read()
	ret, frame2 = cap2.read()
	data = NumpyToJPG(frame1)
	if count == 10:
		clientsocket.sendall(struct.pack("L", len(data))+data)
	data = NumpyToJPG(frame2)
	if count == 10:
		clientsocket.sendall(struct.pack("L", len(data))+data)
		count = 0
	print ("hi %d"%count)
