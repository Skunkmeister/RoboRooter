import cv2
import base64
import roslibpy
import cupy as cp
from CV.imageProcessing import ImageProcessor

def process_topics(handler):
    while(handler.client.is_connected):
        if handler.depth_img is not None and handler.rgb_img is not None:
            handler.ip.process_imgs(handler.depth_img, handler.rgb_img)

class RosHandler:
    def __init__(self):
        self.client = roslibpy.Ros(host='192.168.1.11', port=9090)
        self.ip = ImageProcessor()
        self.rgb_img = None
        self.depth_img = None
        self.ekf_data = None
        self.costmap = None
        self.topic1 = roslibpy.Topic(self.client, '/camera/depth/image_raw', 'sensor_msgs/Image')
        self.topic2 = roslibpy.Topic(self.client, '/camera/rgb/image_raw', 'sensor_msgs/Image')
        self.client.on_ready(self.subscribe_topics(), run_in_thread=True)
        # topic3 = roslibpy.Topic(self.client, '/camera/depth/image_raw', 'sensor_msgs/image_raw')
        # topic4 = roslibpy.Topic(self.client, '/camera/depth/image_raw', 'sensor_msgs/image_raw')

    def subscribe_topics(self)->None:
        self.topic1.subscribe(self.depth_img_process)
        self.topic2.subscribe(self.rgb_img_process)
        # topic3.subscribe(self.ekf_process)
        # topic4.subscribe(self.costmap_proces)
        self.client.run()

    def retrieve_subs(self):
        pass

    def rgb_img_process(self, message)->None:
        img_data = base64.b64decode(message['data'])
        img_array = cp.frombuffer(img_data, dtype=cp.uint8)
        self.rgb_img = cp.reshape(img_array, (message['height'], message['width'], 3))

    def depth_img_process(self, message)->None:
        img_data = base64.b64decode(message['data'])
        img_array = cp.frombuffer(img_data, dtype=cp.uint16)
        self.depth_img = cp.reshape(img_array, (message['height'], message['width']))

    def ekf_process(self, message)->None: # TODO
        pass
    def costmap_process(self, message)->None: # TODO
        pass

    def publish_costmap(self, costmap)->None: # TODO
        pass




