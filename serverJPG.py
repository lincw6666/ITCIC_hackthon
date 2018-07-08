import sys
import cv2
# NO MORE import pickle
from PIL import Image # please pip3 install Pillow
import numpy as np
import struct
import socket
from io import BytesIO

HOST=''
PORT=8089

def JPGToNumpy(jpg):
    file = BytesIO(jpg)
    img = Image.open(file)
    arr = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
    return arr

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print ('Socket created')

s.bind((HOST,PORT))
print ('Socket bind complete')
s.listen(10)
print ('Socket now listening')

conn,addr=s.accept()


data = b''
payload_size = struct.calcsize("L")
while True:
    while len(data) < payload_size:
        data += conn.recv(4096)
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0]
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    ###

    frame=JPGToNumpy(frame_data)
    print(frame)
    cv2.imshow('serverGet',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
