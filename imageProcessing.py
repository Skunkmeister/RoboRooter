import cv2
import numpy as np
from ultralytics import YOLO

def detectObjects(img2D, model, decreaseByProbability, confRequirement):
    results = model.predict(source=img2D, conf=confRequirement, show_labels=False, save = False)
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
        
        mask = results[0].masks.data.numpy()[box]

        if decreaseByProbability:
            decreaseAmount = probabilities[box]
        else:
            decreaseAmount = 1

        for x in range(x1, x2):
            for y in range(y1, y2):
                maskX = x / results[0].orig_shape[1] * results[0].masks.shape[2]
                maskY = y / results[0].orig_shape[0] * results[0].masks.shape[1]
                
                if mask[int(maskY)][int(maskX)] == 1:
                    costmap[y][x] = max(costmap[y][x] - decreaseAmount, 0)
                
    return costmap
    
def depthToWorld(focalLength, cameraXYZ, cameraEuler, depthImage, colorImage):
    # Image dimensions
    [imgHeight, imgWidth] = depthImage.shape[0:2]
    # Initialize matrix to hold world coordinates
    worldCoordinates = np.zeros((imgHeight, imgWidth, 6))
    
    # Convert Euler angles to rotation matrix
    rx = cameraEuler[0]
    ry = cameraEuler[1]
    rz = cameraEuler[2]
    
    Rx = np.matrix([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.matrix([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.matrix([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    
    # The final rotation matrix is Rz * Ry * Rx
    rotationMatrix = Rx
    rotationMatrix = rotationMatrix * Ry
    rotationMatrix = rotationMatrix * Rz
    invRt = np.linalg.inv(rotationMatrix)
    
    # Calculate intrinsic matrix based on focal length and image dimensions
    cx = (imgWidth - 1) / 2  # Assuming the principal point is at the center
    cy = (imgHeight - 1) / 2 # Assuming the principal point is at the center
    intrinsicMatrix = np.matrix([[focalLength, 0, cx],
                       [0, focalLength, cy],
                       [0, 0, 1]])
    
    # Invert the intrinsic matrix for later use
    invIntrinsicMatrix = np.linalg.inv(intrinsicMatrix)
    
    # Loop over each pixel in the depth image
    for y in range(imgHeight):
        for x in range(imgWidth):
            
            depth = -1 * depthImage[y, x, 0]
            pixelCoords = np.matrix([[depth * x], [depth * y], [depth]])
            pixelCoords = invIntrinsicMatrix * pixelCoords
            pixelCoords = invRt * pixelCoords
            
            # Transform to world space
            #worldCoords = tfMatrix * pixelCoords + cameraXYZ
            
            # Store world coordinates
            worldCoordinates[y, x, 0:3] = pixelCoords.transpose() + cameraXYZ
            worldCoordinates[y, x, 3:6] = colorImage[y,x,:]
            points = worldCoordinates.reshape(-1, 6)
    return points
    
#mapPoint format is [y][x][b, g, r, weight]
def flatten(xSize, ySize, scale, points, mapPoints, minimumWeight = None):
    xOffset = xSize / 2
    yOffset = ySize / 2
    
    for point in points:
        xIndex = int((point[0]) * scale + xOffset)
        yIndex = ySize - 1 - int((point[2]) * scale + yOffset)

        if xIndex >= xSize or xIndex < 0 or yIndex >= ySize or yIndex < 0:
            continue
    
        if mapPoints[yIndex][xIndex][3] == 0:
            mapPoints[yIndex][xIndex] = [point[3], point[4], point[5], point[1]]
        else:
            if minimumWeight is not None:
                newWeight = max(point[1], minimumWeight)
                oldWeightTotal = mapPoints[yIndex][xIndex][3]
                newWeightTotal = oldWeightTotal + newWeight
                mapPoints[yIndex][xIndex][3] = newWeightTotal
                mapPoints[yIndex][xIndex][0:3] = (mapPoints[yIndex][xIndex][0:3] * oldWeightTotal + point[3:6] * newWeight) / newWeightTotal
            else:
                n = mapPoints[yIndex][xIndex][3] + 1
                mapPoints[yIndex][xIndex][3] = n
                mapPoints[yIndex][xIndex][0:3] = (mapPoints[yIndex][xIndex][0:3] * (n - 1) + point[3:6]) / n
    return mapPoints

def getImageChunk(environmentMap, costChunkXSize, costChunkYSize, position):
    imageChunk = np.zeros((costChunkYSize, costChunkXSize, 3))
    for y in range(costChunkYSize):
        for x in range(costChunkXSize):
            environmentY = ((position[1] + y) - (costChunkYSize / 2.0)).astype(int)
            environmentX = ((position[0] + x) - (costChunkXSize / 2.0)).astype(int)

            if environmentY >= environmentMap.shape[0] or environmentY < 0 or environmentX >= environmentMap.shape[1] or environmentX < 0:
                continue
            
            imageChunk[y, x, 0:3] = environmentMap[environmentY, environmentX, 0:3]
    return imageChunk

def updateCostMap(costMap, costChunk, position):
    for y in range(costChunk.shape[0]):
        for x in range(costChunk.shape[1]):
            costMapY = ((position[1] + y) - (costChunk.shape[0] / 2.0)).astype(int)
            costMapX = ((position[0] + x) - (costChunk.shape[1] / 2.0)).astype(int)

            if costMapY >= costMap.shape[0] or costMapY < 0 or costMapX >= costMap.shape[1] or costMapX < 0:
                continue
            
            costMap[costMapY, costMapX] = min(costChunk[y, x], costMap[costMapY, costMapX])
    
def main():
    # Settings:
    # Environment Map
    mapXSize = 600
    mapYSize = 800
    mapScale = 80
    # Generate point cloud
    focalLength = 50.0*1920/36.0
    # Flatten
    minimumWeight = 0.25
    # Update Costmap
    costChunkXSize = 300
    costChunkYSize = 200
    model = YOLO('yolov8x-seg.pt')
    decreaseByProbability = True
    confRequirement = 0.01

    # [y][x][b, g, r, weight]
    environmentMap = np.zeros((mapYSize, mapXSize, 4))
    # drivability map equal to environment map
    costMap = np.ones((mapYSize, mapXSize))

    running = True
    while(running):
        # call kalman filter here, should get approximate location and orientation of camera
        cameraXYZ = -1 * np.matrix([5, 5, -2.5])
        cameraXYZ[0, 1] = -1 * cameraXYZ[0, 1]
        cameraEuler = [np.pi/4, 1.22173, 0] # Camera orientation as Euler angles (in radians)

        # Get depth camera current image
        depthImage = "../Depth.exr"
        depthImageArray = cv2.imread(depthImage, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        colorImage = "../Color.png"
        colorImageArray = cv2.imread(colorImage)
        
        # This is a filler location for testing, should be same position from kalman filter
        posMat = np.matrix([300, 300, 0])
        position = [posMat[0, 0], posMat[0, 1]]
    
        # Generate point cloud (may not be needed if depth camera gives automatically, even if so we should include the code but disable it)
        points = depthToWorld(focalLength, cameraXYZ, cameraEuler, depthImageArray, colorImageArray)

        # Flattens point cloud into environmentMap
        flatten(mapXSize, mapYSize, mapScale, points, environmentMap, minimumWeight)

        # Call image stitching Here (will require refactoring of flatten, since needs to be done in the middle of flatten)

        # Get chunk of image for ML Model, right now is implemented as rectangle around rover position
        imageChunk = getImageChunk(environmentMap, costChunkXSize, costChunkYSize, position)

        # Get costmap of chunk
        costChunk = detectObjects(imageChunk, model, decreaseByProbability, confRequirement)

        # Update the costmap
        updateCostMap(costMap, costChunk, position)

        # Here update ROS costmap
        
if __name__ == "__main__":
    main()