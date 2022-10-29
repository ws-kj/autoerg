import cv2
import pytesseract
import os
import sys
import numpy as np
from pytesseract import Output

class Workout(object):

    def __init__(self, path, piece="", time="", dist="", split="", rate="", date=""):
        self.piece = piece
        self.time  = time
        self.dist  = dist
        self.split = split
        self.rate  = rate
        self.date  = date
    
        self.image = cv2.imread(path)
    
    def to_str(self):
        print("\n  Workout: " + self.piece)
        print("  " + self.time + "  " + self.dist + "m  " + self.split)

def proc_image(path):
    img = cv2.imread(path)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img = cv2.fastNlMeansDenoising(img,None,21,7,21)
   
    res = pytesseract.image_to_data(img, output_type=Output.DICT, config="--psm 4")
    
    raw = []
    for i in range(0, len(res["text"])):
        t = res["text"][i]
        if len(t) > 3:
            raw.append(t)

    workout = Workout(path)

    for i in range(0, len(raw)):
        seg = raw[i]
        if workout.piece == "" and seg[0] >= '0' and seg[0] <= '9':
            workout.piece = seg
            continue

        if seg[len(seg)-1] == 'm':
            workout.time = raw[i+1]
            workout.dist = raw[i+2]
            workout.split = raw[i+3]
            break

    print(workout.to_str())            


#    cv2.imshow("erg screen", img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("Processing " + sys.argv[1])
        proc_image(sys.argv[1])    
