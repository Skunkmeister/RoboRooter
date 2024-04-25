#!/usr/bin/env python2.7


import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImageSubscriber:
    def __init__(self):
        rospy.init_node('image_subscriber', anonymous=True)
        self.bridge = CvBridge()
        self.rgb_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.rgb_callback)
        # Replace '/camera/depth/image_raw' with your depth image topic
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)

    def rgb_callback(self, rgb_data):
        cv_image = self.bridge.imgmsg_to_cv2(rgb_data, "rgb8")
        timestamp = rospy.get_rostime().to_sec()
        image_name = f"rgb_image_{timestamp}.jpg"
        cv2.imwrite(image_name, cv_image)
        rospy.loginfo(f"Saved RGB image: {image_name}")

    def depth_callback(self, depth_data):
       cv_image = self.bridge.imgmsg_to_cv2(depth_data, desired_encoding="passthrough")
       timestamp = rospy.get_rostime().to_sec()
       image_name = f"depth_image_{timestamp}.png"  # Depth image can be saved in PNG format
       cv2.imwrite(image_name, cv_image)
       rospy.loginfo(f"Saved depth image: {image_name}")

def main():
    image_subscriber = ImageSubscriber()
    rospy.spin()

if __name__ == '__main__':
    main()