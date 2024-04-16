import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
from ultralytics import YOLO

def detectObjects(img2D, model, decreaseByProbability, confRequirement):
    results = model.predict(source=img2D, conf=confRequirement, show_labels=False, save=False)
    costmap = np.ones((img2D.shape[0], img2D.shape[1]))
    
    numberOfBoxes = results[0].boxes.shape[0]
    boxes = results[0].boxes.numpy()
    probabilities = boxes.conf

    if results[0].masks is None:
        return costmap
    
    correctedBoxes = boxes.xyxy / results[0].orig_shape[1] * results[0].masks.shape[2]
    probMasks = np.multiply(results[0].masks.data.numpy()[:].transpose(), probabilities[:] if decreaseByProbability else 0)
    sumMasks = np.sum(probMasks.transpose(), axis=0)

    costmap = np.clip(costmap - cv2.resize(sumMasks, (img2D.shape[1], img2D.shape[0])), 0, 1)
                
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

    depthImage = -1 * depthImage
    depthImage = depthImage.reshape(-1, 3)

    x, y = np.mgrid[range(imgHeight), range(imgWidth)]
    z = np.ones((imgHeight, imgWidth))
    coords = np.dstack((y, x, z))
    coords = coords.reshape(-1, 3)
    
    coords = np.multiply(coords, depthImage)
    
    coords = invIntrinsicMatrix * coords.transpose()
    coords = invRt * coords
    
    coords = coords.transpose() + cameraXYZ
    
    result = np.zeros((imgWidth * imgHeight, 6))
    result[:, 0:3] = coords
    
    colorImage = colorImage.reshape(-1, 3)
    result[:, 3:6] = colorImage
    
    return result
    
#mapPoint format is [y][x][b, g, r, weight]
def flatten(xSize, ySize, scale, points, mapPoints):
    xOffset = xSize / 2
    yOffset = ySize / 2
    coords = np.stack((points[:, 2] * scale + yOffset, points[:, 0] * scale + xOffset))

    yMin = max(np.min((coords[0, :]).astype(int)), 0)
    yMax = min(np.max((coords[0, :]).astype(int)), ySize)
    yFragSize = yMax - yMin
    xMin = max(np.min((coords[1, :]).astype(int)), 0)
    xMax = min(np.max((coords[1, :]).astype(int)), xSize)
    xFragSize = xMax - xMin

    coords[0, :] -= yMin
    coords[1, :] -= xMin
    
    abs_coords = np.ravel_multi_index(coords.astype(int), (yFragSize, xFragSize), mode = 'clip')
    b = np.bincount(abs_coords, weights = np.multiply(points[:, 3], points[:, 1]), minlength = yFragSize*xFragSize)
    g = np.bincount(abs_coords, weights = np.multiply(points[:, 4], points[:, 1]), minlength = yFragSize*xFragSize)
    r = np.bincount(abs_coords, weights = np.multiply(points[:, 5], points[:, 1]), minlength = yFragSize*xFragSize)
    c = np.bincount(abs_coords, weights = points[:, 1], minlength = yFragSize*xFragSize)
    b = np.divide(b, c, where=c!=0)
    g = np.divide(g, c, where=c!=0)
    r = np.divide(r, c, where=c!=0)
    img = np.dstack((b, g, r))
    img = img.reshape(yFragSize, xFragSize, 3)
    mapPoints = np.flip(mapPoints, 0)
    mask = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
    mask = np.dstack((mask, mask, mask))
    average = (img[:, :] + np.where(mapPoints[yMin:yMax, xMin:xMax] != 0, mapPoints[yMin:yMax, xMin:xMax], img[:, :])) / 2
    mapPoints[yMin:yMax, xMin:xMax] = np.where(mask != 0, average, mapPoints[yMin:yMax, xMin:xMax])
    mapPoints = np.flip(mapPoints, 0)


def getImageChunk(environmentMap, costChunkXSize, costChunkYSize, chunkXOffset, chunkYOffset, position):   
    yMin = (position[1] - (costChunkYSize / 2.0) + chunkYOffset).astype(int)
    yMax = (yMin + costChunkYSize).astype(int)

    yMin = max(0, yMin)
    yMax = min(environmentMap.shape[0], yMax)


    xMin = (position[0] - (costChunkXSize / 2.0) + chunkXOffset).astype(int)
    xMax = (xMin + costChunkXSize).astype(int)

    xMin = max(0, xMin)
    xMax = min(environmentMap.shape[1], xMax)
        
    imageChunk = environmentMap[yMin:yMax, xMin:xMax]
    return imageChunk

def updateCostMap(costMap, costChunk, chunkXOffset, chunkYOffset, position):
    yMin = (position[1] - (costChunk.shape[0] / 2.0) + chunkYOffset).astype(int)
    yMax = (yMin + costChunk.shape[0]).astype(int)

    yMin = max(0, yMin)
    yMax = min(costMap.shape[0], yMax)


    xMin = (position[0] - (costChunk.shape[1] / 2.0) + chunkXOffset).astype(int)
    xMax = (xMin + costChunk.shape[1]).astype(int)

    xMin = max(0, xMin)
    xMax = min(costMap.shape[1], xMax)
        
    costMap[yMin:yMax, xMin:xMax] = costChunk

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
    chunkXOffset = 0
    chunkYOffset = 0
    model = YOLO('yolov8x-seg.pt')
    decreaseByProbability = True
    confRequirement = 0.01

    # [y][x][b, g, r, weight]
    environmentMap = np.zeros((mapYSize, mapXSize, 3))
    # drivability map equal to environment map
    costMap = np.ones((mapYSize, mapXSize))

    running = True
    while(running):
        # TODO GET EKR DATA FOR POSITION AND ORIENTATION HERE, this is hardcoded to example 1 right now
        cameraXYZ = -1 * np.matrix([5, 5, -2.5])
        cameraXYZ[0, 1] = -1 * cameraXYZ[0, 1]
        cameraEuler = [np.pi/4, 1.22173, 0] # Camera orientation as Euler angles (in radians)
        posMat = np.matrix([300, 300, 0])
        position = [posMat[0, 0], posMat[0, 1]]

        # TODO GET DEPTH/COLOR SENSOR DATA, this is hard coded to example file right now
        depthImage = "../Depth.exr"
        depthImageArray = cv2.imread(depthImage, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        colorImage = "../Color.png"
        colorImageArray = cv2.imread(colorImage)
        
    
        # Generate point cloud (may not be needed if depth camera gives automatically, even if so we should include the code but disable it)
        points = depthToWorld(focalLength, cameraXYZ, cameraEuler, depthImageArray, colorImageArray)

        # Flattens point cloud into environmentMap
        flatten(mapXSize, mapYSize, mapScale, points, environmentMap)

        # TODO CALL IMAGE STITCHING HERE (will require refactoring of flatten, since needs to be done in the middle of flatten)

        # Get chunk of image for ML Model, right now is implemented as rectangle around rover position offset by x and y
        imageChunk = getImageChunk(environmentMap, costChunkXSize, costChunkYSize, chunkXOffset, chunkYOffset, position)

        # Get costmap of chunk
        costChunk = detectObjects(imageChunk, model, decreaseByProbability, confRequirement)

        # Update the costmap
        updateCostMap(costMap, costChunk, chunkXOffset, chunkYOffset, position)
        
        # TODO PUBLISH ENVIRONMENT MAP AND COSTMAP TO ROS NODE FOR FRONT END
        
        # TODO UPDATE ROS COSTMAP HERE

        
        print('Loop Finished')
        
if __name__ == "__main__":
    main()