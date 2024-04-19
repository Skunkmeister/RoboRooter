import cv2
import base64
import roslibpy
import cupy as cp
import numpy as np
from CV.imageProcessing import ImageProcessor

def process_topics(handler):
    while(handler.client.is_connected):
        if handler.depth_img is not None and handler.rgb_img is not None and handler.costmap is not None:
            handler.ip.process_imgs(handler.depth_img, handler.rgb_img, handler.mapheight, handler.mapwidth, handler.costmap)

class RosHandler:
    def __init__(self):
        self.client = roslibpy.Ros(host='192.168.1.11', port=9090)
        self.ip = ImageProcessor()
        self.rgb_img = None
        self.depth_img = None
        self.ekf_data = None
        self.costmap = None
        self.mapwidth = 0
        self.mapheight = 0
        self.depth_sub = roslibpy.Topic(self.client, '/camera/depth_registered/image', 'sensor_msgs/Image')
        self.rgb_sub = roslibpy.Topic(self.client, '/camera/rgb/image_raw', 'sensor_msgs/Image')
        # self.ekf_sub = roslibpy.Topic(self.client, '/tf', 'tf2_msgs/TFMessage')
        self.cost_sub = roslibpy.Topic(self.client, '/move_base/local_costmap/costmap', 'nav_msgs/OccupancyGrid')
        self.cost_pub = roslibpy.Topic(self.client, '/move_base/local_costmap/costmap_new', 'nav_msgs/OccupancyGrid')


    def subscribe_topics(self)->None:
        self.depth_sub.subscribe(self.depth_img_process)
        self.rgb_sub.subscribe(self.rgb_img_process)
        # self.ekf_sub.subscribe(self.ekf_process)
        self.cost_sub.subscribe(self.costmap_process)
        self.client.run()

    def is_subbed(self):
        return self.depth_sub.is_subscribed and self.rgb_sub.is_subscribed and self.ekf_sub.is_subscribed and self.cost_sub.is_subscribed

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
    def costmap_process(self, message)->None:
        self.mapwidth = message['info']['width']
        self.mapheight = message['info']['height']
        print(message['data'])
        map_array = cp.array(message['data'])
        self.costmap = cp.reshape(map_array, (self.mapheight, self.mapwidth ))

    def create_occupancy_grid_msg(data, resolution, origin_x, origin_y, origin_theta):
        return {
            'header': {
                'stamp': roslibpy.Time.now(),
                'frame_id': 'map'
            },
            'info': {
                'map_load_time': roslibpy.Time.now(),
                'resolution': resolution,
                'width': data.shape[1],
                'height': data.shape[0],
                'origin': {
                    'position': {
                        'x': origin_x,
                        'y': origin_y,
                        'z': 0
                    },
                    'orientation': roslibpy.helpers.quaternion_from_euler(0, 0, origin_theta)
                }
            },
            'data': data.flatten().tolist()
        }
    def process_topics(self)->None:
        while(self.client.is_connected):
            if self.depth_img is not None and self.rgb_img is not None:
                self.ip.process_imgs(self.depth_img, self.rgb_img,self.mapheight, self.mapwidth, self.costmap)

    def publish_costmap(self, costmap)->None: # TODO
        self.cost_pub.publish(roslibpy.Message(costmap))

    def display_rgb(self, img_type)->None:
        if self.rgb_img is None or self.depth_img is None:
            return
        if img_type == "rgb":
            img_arr = cp.asnumpy(self.depth_img)
            img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        if img_type == "depth":
            img_arr = cp.asnumpy(self.depth_img).astype(np.float32)
            img = cv2.normalize(img_arr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        cv2.imshow('Live Image', img)
        cv2.waitKey(1)

    def kill(self)->None:
        self.depth_sub.unsubscribe()
        self.rgb_sub.unsubscribe()
        # self.ekf_sub.unsubscribe()
        self.cost_sub.unsubscribe()
        self.client.terminate()




