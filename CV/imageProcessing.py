import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import torch
import cv2
import cupy as np
import time
from ultralytics import YOLO

class ImageProcessor:
    def __init__(self):
        self.model = YOLO('yolov8x-seg.pt')# Settings:
        # Environment Map
        self.mapScale = 80
        # Generate point cloud
        self.focalLength = 50.0*1920/36.0
        # Flatten
        self.minimumWeight = 0.25
        # Update Costmap
        self.costChunkXSize = 800
        self.costChunkYSize = 600
        self.chunkXOffset = 0
        self.chunkYOffset = 0
        # model = YOLO('yolov8x-seg.pt')
        self.decreaseByProbability = True
        self.confRequirement = 0.01

    def detectObjects(self, img2D, model, decreaseByProbability, confRequirement, printInfo):
        results = model.predict(source=img2D.get(), conf=confRequirement, show_labels=False, save=False, device='cuda:0', verbose=printInfo)
        costmap = np.ones((img2D.shape[0], img2D.shape[1]))
        
        numberOfBoxes = results[0].boxes.shape[0]
        boxes = results[0].boxes.cpu().numpy()
        probabilities = boxes.conf

        if results[0].masks is None:
            return costmap

        correctedBoxes = boxes.xyxy / results[0].orig_shape[1] * results[0].masks.shape[2]
        if decreaseByProbability:
            probMasks = np.multiply(np.array(results[0].masks.data.cpu().numpy()).transpose(), np.array(probabilities))
        else:
            probMasks = np.array(results[0].masks.data.cpu().numpy()).transpose()

        sumMasks = np.sum(probMasks.transpose(), axis=0)

        costmap = np.clip(costmap - np.array(cv2.resize(np.asnumpy(sumMasks), (img2D.shape[1], img2D.shape[0]))), 0, 1)
                    
        return costmap

    def depthToWorld(self, focalLength, cameraXYZ, cameraEuler, depthImage, colorImage):
        # Image dimensions
        [imgHeight, imgWidth] = depthImage.shape[0:2]
        # Initialize matrix to hold world coordinates
        worldCoordinates = np.zeros((imgHeight, imgWidth, 6))
        
        # Convert Euler angles to rotation matrix
        rx = cameraEuler[0]
        ry = cameraEuler[1]
        rz = cameraEuler[2]
        
        Rx = np.array([[1, 0, 0], [0, np.cos(rx).get(), -np.sin(rx).get()], [0, np.sin(rx).get(), np.cos(rx).get()]])
        Ry = np.array([[np.cos(ry).get(), 0, np.sin(ry).get()], [0, 1, 0], [-np.sin(ry).get(), 0, np.cos(ry).get()]])
        Rz = np.array([[np.cos(rz).get(), -np.sin(rz).get(), 0], [np.sin(rz).get(), np.cos(rz).get(), 0], [0, 0, 1]])
        
        # The final rotation matrix is Rz * Ry * Rx
        rotationMatrix = Rx
        rotationMatrix = np.matmul(rotationMatrix, Ry)
        rotationMatrix = np.matmul(rotationMatrix, Rz)
        invRt = np.linalg.inv(rotationMatrix)
        
        # Calculate intrinsic matrix based on focal length and image dimensions
        cx = (imgWidth - 1) / 2  # Assuming the principal point is at the center
        cy = (imgHeight - 1) / 2 # Assuming the principal point is at the center
        intrinsicMatrix = np.array([[focalLength, 0, cx],
                        [0, focalLength, cy],
                        [0, 0, 1]])
        
        # Invert the intrinsic matrix for later use
        invIntrinsicMatrix = np.linalg.inv(intrinsicMatrix)

        depthImage = -1 * depthImage
        if len(depthImage.shape) < 3 or depthImage.shape[2] == 1:
            depthImage = np.dstack((depthImage, depthImage, depthImage))
        depthImage = depthImage.reshape(-1, 3)

        x, y = np.mgrid[range(imgHeight), range(imgWidth)]
        z = np.ones((imgHeight, imgWidth))
        coords = np.dstack((y, x, z))
        coords = coords.reshape(-1, 3)
        
        coords = np.multiply(coords, depthImage)
        
        coords = np.matmul(invIntrinsicMatrix, coords.transpose())
        coords = np.matmul(invRt, coords)
        
        coords = coords.transpose() + cameraXYZ
        
        result = np.zeros((imgWidth * imgHeight, 6))
        result[:, 0:3] = coords
        
        colorImage = colorImage.reshape(-1, 3)
        result[:, 3:6] = colorImage
        
        return result
        
    def flatten(self, xSize, ySize, scale, points, mapPoints):
        xOffset = xSize / 2
        yOffset = ySize / 2
        coords = np.stack((points[:, 2] * scale + yOffset, points[:, 0] * scale + xOffset))

        yMin = max(np.min((coords[0, :]).astype(int)).item(), 0)
        yMax = min(np.max((coords[0, :]).astype(int)).item(), ySize)
        yFragSize = yMax - yMin
        xMin = max(np.min((coords[1, :]).astype(int)).item(), 0)
        xMax = min(np.max((coords[1, :]).astype(int)).item(), xSize)
        xFragSize = xMax - xMin

        coords[0, :] -= yMin
        coords[1, :] -= xMin

        abs_coords = np.ravel_multi_index(coords.astype(int), (yFragSize, xFragSize), mode = 'clip')
        b = np.bincount(abs_coords, weights = np.multiply(points[:, 3], points[:, 1]), minlength = yFragSize*xFragSize)
        g = np.bincount(abs_coords, weights = np.multiply(points[:, 4], points[:, 1]), minlength = yFragSize*xFragSize)
        r = np.bincount(abs_coords, weights = np.multiply(points[:, 5], points[:, 1]), minlength = yFragSize*xFragSize)
        c = np.bincount(abs_coords, weights = points[:, 1], minlength = yFragSize*xFragSize)
        c = np.where(c!=0, c, np.ones((c.shape)))
        b = np.divide(b, c)
        g = np.divide(g, c)
        r = np.divide(r, c)
        img = np.dstack((b, g, r))
        img = img.reshape(yFragSize, xFragSize, 3)
        mapPoints = np.flip(mapPoints, 0)
        mask = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
        mask = np.dstack((mask, mask, mask))
        average = (img[:, :] + np.where(mapPoints[yMin:yMax, xMin:xMax] != 0, mapPoints[yMin:yMax, xMin:xMax], img[:, :])) / 2
        # Average led to artifacting when not exactly aligned, using overlay instead
        mapPoints[yMin:yMax, xMin:xMax] = np.where(mask != 0, img, mapPoints[yMin:yMax, xMin:xMax])
        mapPoints = np.flip(mapPoints, 0)


    def getImageChunk(self, environmentMap, costChunkXSize, costChunkYSize, chunkXOffset, chunkYOffset, position):   
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

    def updateCostMap(self, costMap, costChunkXSize, costChunkYSize, costChunk, chunkXOffset, chunkYOffset, position):
        yMin = (position[1] - (costChunkYSize / 2.0) + chunkYOffset).astype(int)
        yMax = (yMin + costChunkYSize).astype(int)

        yMin = max(0, yMin)
        yMax = min(costMap.shape[0], yMax)


        xMin = (position[0] - (costChunkXSize / 2.0) + chunkXOffset).astype(int)
        xMax = (xMin + costChunkXSize).astype(int)

        xMin = max(0, xMin)
        xMax = min(costMap.shape[1], xMax)
            
        costMap[yMin:yMax, xMin:xMax] = costChunk

    def process_imgs(self, depthImageArray, colorImageArray, mapYSize, mapXSize, costMap, printInfo=True):
        # [y][x][b, g, r, weight]
        environmentMap = np.zeros((mapYSize, mapXSize, 3))
        # drivability map equal to environment map
        # print(f"RGB Image: {colorImageArray}")
        print(f"Depth Image: {depthImageArray}")
        # print(f"Map Dimensions: {mapYSize} , {mapXSize}")
        # print(f"{costMap}")

        time1 = time.time()
        
        # TODO GET EKR DATA FOR POSITION AND ORIENTATION HERE, this is hardcoded to example 1 right now
        cameraXYZ = -1 * np.array([5, 5, -2.5])
        cameraXYZ[1] = -1 * cameraXYZ[1]
        cameraEuler = [np.pi/4, 1.22173, 0] # Camera orientation as Euler angles (in radians)
        posMat = np.array([mapXSize / 2, mapYSize / 2, 0])
        position = [posMat[0], posMat[1]]
    
        # Generate point cloud (may not be needed if depth camera gives automatically, even if so we should include the code but disable it)
        points = self.depthToWorld(self.focalLength, cameraXYZ, cameraEuler, depthImageArray, colorImageArray)

        # Flattens point cloud into environmentMap
        self.flatten(mapXSize, mapYSize, self.mapScale, points, environmentMap)

        # TODO CALL IMAGE STITCHING HERE (will require refactoring of flatten, since needs to be done in the middle of flatten)

        # Get chunk of image for ML Model, right now is implemented as rectangle around rover position offset by x and y
        imageChunk = self.getImageChunk(environmentMap, self.costChunkXSize, self.costChunkYSize, self.chunkXOffset, self.chunkYOffset, position)

        # Get costmap of chunk
        costChunk = self.detectObjects(imageChunk, self.model, self.decreaseByProbability, self.confRequirement, printInfo)

        # Update the costmap
        self.updateCostMap(costMap, self.costChunkXSize, self.costChunkYSize, costChunk, self.chunkXOffset, self.chunkYOffset, position)
        

        time2 = time.time()
        
        if printInfo:
            newTime = (time2 - time1)*1000
            print('Time'+ ': ' + str(newTime)[0:5] + ' ms')
            
            
