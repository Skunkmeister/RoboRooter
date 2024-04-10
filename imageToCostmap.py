import numpy as np
def detectObjects(img2D, model, decreaseByProbability):
    results = model.predict(source=img2D, conf=0.01, show_labels=False, save = True)
    costmap = np.ones((img2D.shape[0], img2D.shape[1]))
    
    numberOfBoxes = results[0].boxes.shape[0]
    boxes = results[0].cpu().boxes.numpy()
    probabilities = boxes.conf
    
    for box in range(numberOfBoxes):
        currBox = boxes.xyxy[box]
        x1 = currBox[0].astype(int)
        x2 = currBox[2].astype(int)
        y1 = currBox[1].astype(int)
        y2 = currBox[3].astype(int)
        
        mask = results[0].cpu().masks.data.numpy()[box]

        if decreaseByProbability:
            decreaseAmount = probabilities[box]
        else:
            decreaseAmount = 1

        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                maskX = x / results[0].orig_shape[1] * results[0].masks.shape[2]
                maskY = y / results[0].orig_shape[0] * results[0].masks.shape[1]
                
                if mask[int(maskY)][int(maskX)] == 1:
                    costmap[y][x] = max(costmap[y][x] - decreaseAmount, 0)
                
    return costmap
