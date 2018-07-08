import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import scipy.misc
import pyttsx3
import cv2
import pipes
import subprocess
import sys
import cv2
import pickle
from PIL import Image # please pip3 install Pillow
import numpy as np
import struct
import socket
from io import BytesIO

from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

#cap1 = cv2.VideoCapture(1)
#cap2 = cv2.VideoCapture(2)

#if not cap1.isOpened() or not cap2.isOpened():
#  print("Can't open camera!")
#  exit()

#cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
#cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
#cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

# ++++++++++++++++++++++++++++++++++


HOST='192.168.43.137'
PORT=8000

def JPGToNumpy(jpg):
    file = np.asarray(bytearray(jpg), np.uint8)
    img = cv2.imdecode(file, 1)
    return img

def PickleToNumpy(pic):
    return pickle.loads(pic)

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print ('Socket created')

s.bind((HOST,PORT))
print ('Socket bind complete')
s.listen(10)
print ('Socket now listening')

conn,addr=s.accept()


data = b''
payload_size = struct.calcsize("L")
#++++++++++++++++++++++++++++++++++++++++

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)
#for file in tar_file.getmembers():
#  file_name = os.path.basename(file.name)
#  if 'frozen_inference_graph.pb' in file_name:
#    tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def obj_detection(img, process, GOIN, isLeft):
  image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
  detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
  detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
  detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
  num_detections = detection_graph.get_tensor_by_name('num_detections:0')
  image_np_expanded = np.expand_dims(img, axis=0)

  (boxes, scores, classes, num) = sess.run(
      [detection_boxes, detection_scores, detection_classes, num_detections],
      feed_dict={image_tensor: image_np_expanded})
  width = img.shape[0]
  height = img.shape[1]
  for get_data in zip(classes[0], scores[0], boxes[0]):
    ymin = get_data[2][0]*height
    xmin = get_data[2][1]*width
    ymax = get_data[2][2]*height
    xmax = get_data[2][3]*width
    obj = str(get_data[0])
    if 1 <= get_data[0] <= 4 and get_data[1] > 0.5 and GOIN == True and (xmax - xmin) >= 50 and (ymax - ymin) >= 50:
      print("x: ",xmax - xmin)
      print("y: ",ymax - ymin)
      if isLeft:
        process = subprocess.Popen(["python", "googleSpeech.py", obj, "2"])
        conn.send((obj+",2").encode())
      else:
        process = subprocess.Popen(["python", "googleSpeech.py", obj, "1"])
        conn.send((obj+",1").encode())
      GOIN = False
      break
  else:
    conn.send("0,0".encode())
  if GOIN == False:
    if process.poll() != None:
      GOIN = True
#
  vis_util.visualize_boxes_and_labels_on_image_array(
      img,
      np.squeeze(boxes),
      np.squeeze(classes).astype(np.int32),
      np.squeeze(scores),
      category_index,
      use_normalized_coordinates=True,
      line_thickness=4)

  return img, process, GOIN

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    GOIN1 = True
    GOIN2 = True
    process1 = 0
    process2 = 0
    while True:
      # Get Frame One
      frame = None
      while len(data) < payload_size:
          data += conn.recv(409600)
      packed_msg_size = data[:payload_size]
      data = data[payload_size:]
      msg_size = struct.unpack("L", packed_msg_size)[0]
      while len(data) < msg_size:
          data += conn.recv(409600)
      frame_data = data[:msg_size]
      data = data[msg_size:]
      ###
      frame=JPGToNumpy(frame_data)
      image_np1 = np.array(frame)
      img1, process1, GOIN1 = obj_detection(image_np1, process1, GOIN1, True)

      # Get Frame Two
      frame = None
      while len(data) < payload_size:
          data += conn.recv(409600)
      packed_msg_size = data[:payload_size]
      data = data[payload_size:]
      msg_size = struct.unpack("L", packed_msg_size)[0]
      while len(data) < msg_size:
          data += conn.recv(409600)
      frame_data = data[:msg_size]
      data = data[msg_size:]
      ###
      frame=JPGToNumpy(frame_data)
      image_np2 = np.array(frame)
      img2, process2, GOIN2 = obj_detection(image_np2, process2, GOIN2, False)

      cv2.imshow('object detection1', img1)
      cv2.imshow('object detection2', img2)
      k = cv2.waitKey(33)
      if k==27:    # Esc key to stop
        print("hehehe\n")
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
        break
