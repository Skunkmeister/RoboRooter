import cv2
import base64
import roslibpy
import cupy as cp
import numpy as np
from CV.imageProcessing import ImageProcessor

def process_topics(hndlr)->None:
        while(hndlr.client.is_connected):
            if all(x is not None for x in [hndlr.depth_img, hndlr.rgb_img, hndlr.costmap, hndlr.translation, hndlr.rotation]):
                hndlr.ip.process_imgs(hndlr, hndlr.depth_img, hndlr.rgb_img,hndlr.mapheight, 
                                     hndlr.mapwidth, hndlr.costmap, hndlr.translation, hndlr.rotation)

class RosHandler:
    def __init__(self):
        self.client = roslibpy.Ros(host='192.168.1.11', port=9090)
        self.ip = ImageProcessor()
        self.rgb_img = None
        self.depth_img = None
        self.costmap = None
        self.translation = None
        self.rotation = None
        self.mapwidth = 0
        self.mapheight = 0
        self.depth_sub = roslibpy.Topic(self.client, '/camera/depth_registered/image', 'sensor_msgs/Image')
        self.rgb_sub = roslibpy.Topic(self.client, '/camera/rgb/image_raw', 'sensor_msgs/Image')
        self.tf_sub = roslibpy.Topic(self.client, '/tf', 'tf2_msgs/TFMessage')
        self.cost_sub = roslibpy.Topic(self.client, '/move_base/local_costmap/pre_costmap', 'nav_msgs/OccupancyGrid')
        self.cost_pub = roslibpy.Topic(self.client, '/move_base/local_costmap/costmap', 'nav_msgs/OccupancyGrid')


    def subscribe_topics(self)->None:
        self.depth_sub.subscribe(self.depth_img_process)
        self.rgb_sub.subscribe(self.rgb_img_process)
        self.tf_sub.subscribe(self.tf_process)
        self.cost_sub.subscribe(self.costmap_process)
        self.client.run()

    def rgb_img_process(self, message)->None:
        img_data = base64.b64decode(message['data'])
        img_array = cp.frombuffer(img_data, dtype=cp.uint8)
        self.rgb_img = cp.reshape(img_array, (message['height'], message['width'], 3))

    def depth_img_process(self, message)->None:
        img_data = base64.b64decode(message['data'])
        img_array = cp.frombuffer(img_data, dtype=cp.uint16)
        self.depth_img = cp.reshape(img_array, (message['height'], message['width']))

    def tf_process(self, message)->None:
        for transform in message['transforms']:
            if transform['header']['frame_id'] == '/base_link' and transform['child_frame_id'] == '/camera':
                translation = transform['transform']['translation']
                rotation = transform['transform']['rotation']
                self.translation = cp.array([translation['x'], translation['y'], translation['z']])
                self.rotation = cp.array([rotation['x'], rotation['y'], rotation['z'], rotation['w']])

    def costmap_process(self, message)->None:
        if message is not None:
            print(message)
            self.mapwidth = message['info']['width']
            self.mapheight = message['info']['height']
            map_array = cp.array(message['data'])
            self.costmap = cp.reshape(map_array, (self.mapheight, self.mapwidth ))

    def create_occgrid_msg(self, data, resolution, origin_x, origin_y, origin_theta, frame_id='odom'):
        """
        Args:
        data (np.ndarray): 2D numpy array representing the occupancy grid.
        resolution (float): Map resolution in meters per pixel.
        origin_x (float): Origin x-coordinate in meters.
        origin_y (float): Origin y-coordinate in meters.
        origin_theta (float): Origin orientation in radians.
        frame_id (str): The coordinate frame ID to which the map is referenced.

        Returns:
        dict: Properly formatted dictionary for OccupancyGrid messages.
        """
        current_time = roslibpy.Time.now()
        # quaternion = roslibpy.helpers.quaternion_from_euler(0, 0, origin_theta)

        return {
            'header': {
                'seq': 0,  # Sequence number (you may want to manage this externally if needed)
                'stamp': {
                    'secs': current_time.secs,
                    'nsecs': current_time.nsecs
                },
                'frame_id': frame_id
            },
            'info': {
                'map_load_time': {
                    'secs': current_time.secs,
                    'nsecs': current_time.nsecs
                },
                'resolution': resolution,
                'width': 60,
                'height': 60,
                'origin': {
                    'position': {
                        'x': origin_x,
                        'y': origin_y,
                        'z': 0.0  
                    },
                    'orientation': {
                        'x': 0.0,
                        'y': 0.0,
                        'z': 0.0,
                        'w': 1.0
                    }
                }
            },
            'data': data.flatten().tolist()  
        }

    def publish_costmap(self, costmap)->None: # TODO
        self.cost_pub.publish(roslibpy.Message(costmap))

    # used for testing to see if live images are being received & stored correctly
    def display_rgb(self, img_type="rgb")->None: 
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

    def is_subbed(self)->bool:
        return self.depth_sub.is_subscribed and self.rgb_sub.is_subscribed and self.tf_sub.is_subscribed and self.cost_sub.is_subscribed
    
    def kill(self)->None:
        self.depth_sub.unsubscribe()
        self.rgb_sub.unsubscribe()
        self.tf_sub.unsubscribe()
        self.cost_sub.unsubscribe()
        self.client.terminate()