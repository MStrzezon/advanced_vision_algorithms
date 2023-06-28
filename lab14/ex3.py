import cv2
import numpy as np
from matplotlib import pyplot as plt

events_filepath = 'dataset/events.txt'

events = []

with open(events_filepath, 'r') as f:
    while True:
        line = f.readline()
        if not line or float(line.split(' ')[0]) >= 2:
            break
        if float(line.split(' ')[0]) > 1:
            events.append(line.split(' '))



def event_frame(x, y, polarity, shape):
    matrix = np.ones(shape)
    matrix = matrix * 127
    matrix = matrix.astype(np.uint8)
    for i in range(len(x)):
        if polarity[i] == 1:
            matrix[y[i]][x[i]] = 255
        else:
            matrix[y[i]][x[i]] = 0
    return matrix

tau = 0.01

reset = True
for event in events:
    if reset:
        x = []
        y = []
        polarity = []
        first_timestamp = float(event[0])
        last_timestamp = float(event[0])
        reset = False
    if last_timestamp - first_timestamp >= tau:
        img = event_frame(x, y, polarity, (180, 250))
        cv2.imshow('frame', img)
        cv2.waitKey(0)
        reset = True
        continue
    last_timestamp = float(event[0])
    x.append(int(event[1]))
    y.append(int(event[2]))
    polarity.append(int(event[3]) if float(event[3]) == 1 else -1)
cv2.destroyAllWindows()