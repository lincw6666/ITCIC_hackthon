import os
import sys
import pyttsx3

MODE = 0
SIDE = 0

def alert(mode, side):
    engine = pyttsx3.init()
    object = ""
    msg = ""
    print("mode: " + mode + ", side: " + side)
    if mode == '1': object = "人"
    if mode == '2': object = "腳踏車"
    if mode == '3': object = "汽車"
    if mode == '4': object = "機車"
    if side == '1': msg = '右邊有' + object + '唷寶貝！小心小心！'
    if side == '2': msg = '左邊有' + object + '唷寶貝！小心小心！'
    engine.say(msg)
    print(msg)
    engine.runAndWait()

alert(sys.argv[1], sys.argv[2])
