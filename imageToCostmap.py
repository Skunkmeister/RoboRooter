import numpy as np
from PIL import Image

def detectObjects(img2D, model, decreaseByProbability):
    results = model.predict(source=img2D, conf=0.01)
    costmap = np.ones((img2D.shape[0], img2D.shape[1]))
    
    numberOfBoxes = results[0].boxes.shape[0]
    boxes = results[0].boxes.numpy()
    probabilities = boxes.conf
    
    for box in range(numberOfBoxes):
        currBox = boxes.xyxy[box]
        x1 = currBox[0].astype(int)
        x2 = currBox[2].astype(int)
        y1 = currBox[1].astype(int)
        y2 = currBox[3].astype(int)

        if decreaseByProbability:
            decreaseAmount = probabilities[box]
        else:
            decreaseAmount = 1

        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                costmap[y][x] -= decreaseAmount
                
    return costmap
