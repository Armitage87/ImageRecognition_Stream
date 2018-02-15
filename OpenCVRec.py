import numpy as np
import argparse
import time
from PIL import Image
import cv2

cap = cv2.VideoCapture(0)

while (1):

    _, image = cap.read()
    rows = open(".\\synset_words.txt").read().strip().split("\n")
    classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

    blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

    #print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(".\\bvlc_googlenet.prototxt", ".\\bvlc_googlenet.caffemodel")

    net.setInput(blob)
    start = time.time()
    preds = net.forward()
    end = time.time()
    #print("[INFO] cassification took {:.5} seconds".format(end-start))

    idxs = np.argsort(preds[0])[::-1][:5]

    for (i, idx) in enumerate(idxs):
        if i == 0:
            text = "label: {}, {:.2f}%".format(classes[idx], preds[0][idx] * 100)
            cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print("[INFO] {}, label: {}, probability: {:.5}".format(i + 1, classes[idx], preds[0][idx]))

    cv2.imshow("image", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
