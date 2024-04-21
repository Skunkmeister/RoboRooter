#!/usr/bin/env python
# coding:utf-8
import tf
import rospy
import numpy as np
import dynamic_reconfigure.client
from common import *
from time import sleep
from transbot_msgs.msg import JoyState
from sensor_msgs.msg import LaserScan
from math import radians, copysign, sqrt, pow
from geometry_msgs.msg import Twist, Point, Quaternion
from transform_utils import quat_to_angle, normalize_angle
from transbot_bringup.cfg import PatrolParamConfig
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from dynamic_reconfigure.server import Server
from transbot_laser.cfg import laserAvoidPIDConfig

class TransbotPatrol():
    def __init__(self):
        rospy.on_shutdown(self.cancel)
        self.r = rospy.Rate(20)
        self.moving = True
        self.Joy_active = False
        self.SetLoop = False
        self.Switch = False
        self.command_src = "finish"
        self.Command = "finish"
        self.reverse = 1
        self.warning = 1
        self.Length = 1.0
        self.Angle = 360
        self.ResponseDist = 0.7
        self.LaserAngle = 20
        self.Linear = 0.3
        self.Angular = 1.0
        self.RotationScaling = 1.0
        self.LineScaling = 1.0
        self.LineTolerance = 0.1
        self.RotationTolerance = 0.3
        self.tf_listener = tf.TransformListener()
        self.odom_frame = rospy.get_param('~odom_frame', '/odom')
        self.base_frame = rospy.get_param('~base_frame', '/base_footprint')
        Server(PatrolParamConfig, self.dynamic_reconfigure_callback)
        self.pub_cmdVel = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.registerScan)
        self.sub_Joy = rospy.Subscriber('/JoyState', JoyState, self.JoyStateCallback)
        self.tf_listener.waitForTransform(self.odom_frame, self.base_frame, rospy.Time(), rospy.Duration(60.0))
        self.dyn_client = dynamic_reconfigure.client.Client("TransbotPatrol", timeout=60)
        rospy.loginfo("Bring up rqt_reconfigure to control the Transbot.")

    def dynamic_reconfigure_callback(self, config, level):
        '''
        Patrol gameplay settings 巡逻玩法设置
        :param Commond: 巡逻指令 [LengthTest, AngleTest, Triangle, Square, Parallelogram, Circle]
        :param Length: 运行长度（米）[0.3,3]
        :param Linear: 运行速度（米/秒）[0.2,1.2]
        :param SetLoop: 是否循环巡逻 [True or False]
        :param ResponseDist: 激光雷达避障距离[0.4,8]
        :param LaserAngle: 激光雷达避障角度[10,180]
        :param LineScaling: 直线距离缩放比例
        :param RotationScaling: 旋转角度缩放比例
        :param LineTolerance: 允许的直线距离误差
        :param RotationTolerance: 允许的旋转角度误差
        :param Switch: 巡逻功能【开始/暂停】
        '''
        self.Linear = config['Linear']
        self.Angular = config['Angular']
        self.Length = config['Length']
        self.Angle = config['Angle']
        self.ResponseDist = config['ResponseDist']
        self.LaserAngle = config['LaserAngle']
        self.LineScaling = config['LineScaling']+0.08
        self.RotationScaling = config['RotationScaling']+0.2
        self.LineTolerance = config['LineTolerance']
        self.RotationTolerance = config['RotationTolerance']
        self.Command = config['Command']
        self.command_src = config['Command']
        self.SetLoop = config['SetLoop']
        self.Switch = config['Switch']
        if self.Switch == True: print ("Command: ", self.Command)
        return config

    def Triangle(self, index, angle):
        index += 1
        advancing = False
        sleep(0.5)
        if index == 1 or index == 3: advancing = self.advancing(self.Length)
        elif index == 2:
            side = sqrt(pow((self.Length), 2) * 2)
            advancing = self.advancing(side)
        sleep(0.5)
        if advancing == True:
            spin = self.Spin(angle)
            if spin == True:
                if index == 1:
                    self.Triangle(index, 135)
                elif index == 2:
                    self.Triangle(index, 90)
                else:
                    self.Command = "finish"
                    return

    def Square(self, index, angle):
        index += 1
        if index == 5:
            self.Command = "finish"
            return
        sleep(0.5)
        advancing = self.advancing(self.Length)
        sleep(0.5)
        print ("advancing: ",advancing)
        if advancing == True:
            spin = self.Spin(angle)
            if spin == True:
                if index == 2:
                    self.Square(index, 90)
                else:
                    self.Square(index, 90)

    def Parallelogram(self, index, angle):
        index += 1
        if index == 5:
            self.Command = "finish"
            return
        sleep(0.5)
        advancing = self.advancing(self.Length)
        sleep(0.5)
        if advancing == True:
            spin = self.Spin(angle)
            if spin == True:
                if index == 2:
                    self.Parallelogram(index, 120)
                else:
                    self.Parallelogram(index, 60)

    def process(self):
        index = 0
        while not rospy.is_shutdown():
            if self.Switch==True:
                if self.Command == "LengthTest":
                    advancing = self.advancing(self.Length)
                    if advancing == True: self.Command = "finish"
                elif self.Command == "AngleTest":
                    spin = self.Spin(self.Angle)
                    if spin == True: self.Command = "finish"
                elif self.Command == "Triangle": self.Triangle(index, 135)
                elif self.Command == "Square": self.Square(index, 90)
                elif self.Command == "Parallelogram": self.Parallelogram(index, 120)
                elif self.Command == "Circle":
                    spin = self.Spin(360)
                    if spin == True: self.Command = "finish"
                if self.Command == "finish":
                    self.pub_cmdVel.publish(Twist())
                    if self.SetLoop == False:
                        params = {'Switch': False}
                        self.dyn_client.update_configuration(params)
                    else: self.Command = self.command_src
            self.r.sleep()
        self.pub_cmdVel.publish(Twist())

    def JoyStateCallback(self, msg):
        if not isinstance(msg, JoyState): return
        self.Joy_active = msg.state
        if not self.Joy_active: self.pub_cmdVel.publish(Twist())

    def registerScan(self, scan_data):
        if self.ResponseDist == 0 or self.moving == False: return
        # 记录激光扫描并发布最近物体的位置（或指向某点）
        # registers laser scan and publishes position of closest object (or point rather)
        ranges = np.array(scan_data.ranges)
        # 按距离排序以检查从较近的点到较远的点是否是真实的东西
        # sort by distance to check from closer to further away points if they might be something real
        sortedIndices = np.argsort(ranges)
        # print ("laser_depth: ",len(sortedIndices))
        self.warning = 1
        # if we already have a last scan to compare to:
        for i in sortedIndices:
            if len(np.array(scan_data.ranges)) == 720:
                # 通过清除不需要的扇区的数据来保留有效的数据
		# Keep valid data by clearing the data of unnecessary sectors
                if i in range(self.LaserAngle * 2, 720 - self.LaserAngle * 2): continue
                else:
                    if ranges[i] < self.ResponseDist:
                        self.warning += 1
                        print ("i: {},ranges[i]: {},warning: {}".format(i, ranges[i], self.warning))
            elif len(np.array(scan_data.ranges)) == 360:
		# Keep valid data by clearing the data of unnecessary sectors
                # 通过清除不需要的扇区的数据来保留有效的数据
                if i in range(self.LaserAngle, 360 - self.LaserAngle): continue
                else:
                    if ranges[i] < self.ResponseDist:
                        self.warning += 1
                        print ("i: {},ranges[i]: {},warning: {}".format(i, ranges[i], self.warning))
        # print ("warning: {}".format(self.warning))

    def Spin(self, angle):
        target_angle = radians(angle)
        odom_angle = self.get_odom_angle()
        last_angle = odom_angle
        turn_angle = 0
        # Alternate directions between tests
        while not rospy.is_shutdown():
            # Get the current rotation angle from tf
            odom_angle = self.get_odom_angle()
            # Compute how far we have gone since the last measurement
            delta_angle = self.RotationScaling * normalize_angle(odom_angle - last_angle)
            # Add to our total angle so far
            turn_angle += delta_angle
            # Compute the new error
            error = target_angle - turn_angle
            # Store the current angle for the next comparison
            last_angle = odom_angle
            print("self.RotationScaling: {},target_angle: {},turn_angle: {},error: {}".format(
                self.RotationScaling, target_angle, turn_angle, abs(error)))
            move_cmd = Twist()
            if (abs(error) < self.RotationTolerance) or self.Switch == False :
                self.pub_cmdVel.publish(Twist())
                return True
            if self.Joy_active or self.warning > 10 :
                if self.moving == True:
                    self.pub_cmdVel.publish(Twist())
                    self.moving = False
                continue
            else:
                if self.Command == "Circle":
                    adjust = 6.0 / 5.0
                    length = self.Linear * adjust / self.Length
                    move_cmd.linear.x = self.Linear
                    move_cmd.angular.z = copysign(length, error)
                else: move_cmd.angular.z = copysign(self.Angular, error)
                self.pub_cmdVel.publish(move_cmd)
            self.moving = True
            self.r.sleep()
        # Stop the robot
        self.pub_cmdVel.publish(Twist())
        return True

    def advancing(self, target_distance):
        position = self.get_position()
        x_start, y_start = position.x, position.y
        print ("x_start: {}, y_start: {}".format(x_start, y_start))
        while not rospy.is_shutdown():
            position = self.get_position()
            # Compute the Euclidean distance from the target point
            distance = sqrt(pow((position.x - x_start), 2) + pow((position.y - y_start), 2))
            # rospy.loginfo(position)
            distance *= self.LineScaling
            # How close are we?
            error = distance - target_distance
            print ("target_distance: {},distance: {}".format(target_distance, distance))
            # If not, move in the appropriate direction
            move_cmd = Twist()
            move_cmd.linear.x = self.Linear
            if abs(error) <= self.LineTolerance or self.Switch == False:
                self.pub_cmdVel.publish(Twist())
                return True
            if self.Joy_active or self.warning > 10:
                if self.moving == True:
                    self.pub_cmdVel.publish(Twist())
                    self.moving = False
                continue
            else:
                self.pub_cmdVel.publish(move_cmd)
            self.moving = True
        self.pub_cmdVel.publish(Twist())
        return False

    def get_odom_angle(self):
        # Get the current transform between the odom and base frames
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return
        # Convert the rotation from a quaternion to an Euler angle
        return quat_to_angle(Quaternion(*rot))

    def get_position(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return
        return Point(*trans)

    def cancel(self):
        # Always stop the robot when shutting down the node
        rospy.loginfo("Stopping the robot...")
        self.pub_cmdVel.publish(Twist())
        self.pub_cmdVel.unregister()
        self.sub_scan.unregister()
        self.sub_Joy.unregister()
        self.tf_listener.clear()
        rospy.sleep(1)

class laserAvoid:
    def __init__(self):
        rospy.on_shutdown(self.cancel)
        self.r = rospy.Rate(20)
        self.linear = 0.3
        self.angular = 1
        self.ResponseDist = 0.55
        self.Length = 10 #EDIT THIS MANUALLY
        self.Linear = 0.3 #EDIT THIS MANUALLY
        self.Switch = False #EDIT THIS MANUALLY
        self.RotationScaling = 1.0 #EDIT THIS MANUALLY
        self.RotationTolerance = 0.3 #EDIT THIS MANUALLY
        self.pub_cmdVel = rospy.Publisher('/cmd_vel', Twist, queue_size=5) #EDIT THIS MANUALLY
        self.LaserAngle = 30  # 10~180
        self.Moving = False
        self.switch = False
        self.running = False
        self.Right_warning = 0
        self.Left_warning = 0
        self.front_warning = 0
        self.ros_ctrl = ROSCtrl()
        self.sub_laser = rospy.Subscriber('/scan', LaserScan, self.registerScan)
        Server(laserAvoidPIDConfig, self.dynamic_reconfigure_callback)


    def cancel(self):
        self.ros_ctrl.cancel()
        self.sub_laser.unregister()
        rospy.loginfo("Shutting down this node.")

    def Spin(self, angle):
        target_angle = radians(angle)
        odom_angle = self.get_odom_angle()
        last_angle = odom_angle
        turn_angle = 0
        # Alternate directions between tests
        while not rospy.is_shutdown():
            # Get the current rotation angle from tf
            odom_angle = self.get_odom_angle()
            # Compute how far we have gone since the last measurement
            delta_angle = self.RotationScaling * normalize_angle(odom_angle - last_angle)
            # Add to our total angle so far
            turn_angle += delta_angle
            # Compute the new error
            error = target_angle - turn_angle
            # Store the current angle for the next comparison
            last_angle = odom_angle
            print("self.RotationScaling: {},target_angle: {},turn_angle: {},error: {}".format(
                self.RotationScaling, target_angle, turn_angle, abs(error)))
            move_cmd = Twist()
            if (abs(error) < self.RotationTolerance) or self.Switch == False :
                self.pub_cmdVel.publish(Twist())
                return True
            if self.Joy_active or self.warning > 10 :
                if self.moving == True:
                    self.pub_cmdVel.publish(Twist())
                    self.moving = False
                continue
            else:
                if self.Command == "Circle":
                    adjust = 6.0 / 5.0
                    length = self.Linear * adjust / self.Length
                    move_cmd.linear.x = self.Linear
                    move_cmd.angular.z = copysign(length, error)
                else: move_cmd.angular.z = copysign(self.Angular, error)
                self.pub_cmdVel.publish(move_cmd)
            self.moving = True
            self.r.sleep()
        # Stop the robot
        self.pub_cmdVel.publish(Twist())
        return True

    def dynamic_reconfigure_callback(self, config, level):
        self.switch = config['switch']
        self.linear = config['linear']
        self.angular = config['angular']
        self.LaserAngle = config['LaserAngle']
        self.ResponseDist = config['ResponseDist']
        return config

    def registerScan(self, scan_data):
        if self.running == True: return
        # 记录激光扫描并发布最近物体的位置（或指向某点）
        # Record the laser scan and publish the position of the nearest object (or point to a point)
        ranges = np.array(scan_data.ranges)
        # 按距离排序以检查从较近的点到较远的点是否是真实的东西
        # Sort by distance to check whether things are real from closer points to more distant points
        sortedIndices = np.argsort(ranges)
        self.Right_warning = 0
        self.Left_warning = 0
        self.front_warning = 0
        #print "scan_data:", len(sortedIndices)
        # if we already have a last scan to compare to:
        for i in sortedIndices:
            if len(np.array(scan_data.ranges)) == 720:
                # 通过清除不需要的扇区的数据来保留有效的数据
                # Keep valid data by purging data from unneeded sectors
                if 20 < i < self.LaserAngle * 2:
                    if ranges[i] < self.ResponseDist: self.Left_warning += 1
                elif (720 - self.LaserAngle * 2) < i < 700:
                    if ranges[i] < self.ResponseDist: self.Right_warning += 1
                elif (700 <= i ) or ( i <= 20):
                    if ranges[i] <= self.ResponseDist: self.front_warning += 1
            elif len(np.array(scan_data.ranges)) == 360:
                # 通过清除不需要的扇区的数据来保留有效的数据
                # Keep valid data by purging data from unneeded sectors
                if 10 < i < self.LaserAngle:
                    if ranges[i] < self.ResponseDist: self.Left_warning += 1
                elif (350 - self.LaserAngle) < i < 350:
                    if ranges[i] < self.ResponseDist: self.Right_warning += 1
                elif (350 <= i <= 360) or (0<= i <=10):
                    # print ("i: {},dist: {}", format(i, ranges[i]))
                    if ranges[i] < self.ResponseDist: self.front_warning += 1
        # print (self.Left_warning,self.front_warning,self.Right_warning)


    def robot_move(self):
        while not rospy.is_shutdown():
            if self.ros_ctrl.Joy_active or self.switch == True:
                if self.Moving == True:
                    self.ros_ctrl.pub_vel.publish(Twist())
                    self.Moving = not self.Moving
                continue
            self.Moving = True
            twist = Twist()
            # 左正右负
            # Left positive and right negative
            if self.front_warning > 10 and self.Left_warning > 10 and self.Right_warning > 10:
                # print ('1、左右中有障碍物，右转')
                # print ('1, there are obstacles in the left and right, turn right')
                twist.linear.x = -0.15
                twist.angular.z = -self.angular
                self.ros_ctrl.pub_vel.publish(twist)
                sleep(0.2)
            elif self.front_warning > 10 and self.Left_warning <= 10 and self.Right_warning > 10:
                # print ('2、右中有障碍物，左转')
                # print ('2, there is an obstacle in the middle right, turn left')
                twist.linear.x = 0
                twist.angular.z = self.angular
                self.ros_ctrl.pub_vel.publish(twist)
                sleep(0.2)
                if self.Left_warning > 10 and self.Right_warning <= 10:
                    # print ('3、左有障碍物，右转')
                    # print ('3, there is an obstacle on the left, turn right')
                    twist.linear.x = 0
                    twist.angular.z = -self.angular
                    self.ros_ctrl.pub_vel.publish(twist)
                    sleep(0.4)
            elif self.front_warning > 10 and self.Left_warning > 10 and self.Right_warning <= 10:
                # print ('4、左中有障碍物，右转')
                # print ('4. There is an obstacle in the middle left, turn right')
                twist.linear.x = 0
                twist.angular.z = -self.angular
                self.ros_ctrl.pub_vel.publish(twist)
                sleep(0.2)
                if self.Left_warning <= 10 and self.Right_warning > 10:
                    # print ('5、左有障碍物，左转')
                    # print ('5, there is an obstacle on the left, turn left')
                    twist.linear.x = 0
                    twist.angular.z = self.angular
                    self.ros_ctrl.pub_vel.publish(twist)
                    sleep(0.4)
            elif self.front_warning > 10 and self.Left_warning < 10 and self.Right_warning < 10:
                # print ('6、中有障碍物，左转')
                # print ('6, there is an obstacle in the middle, turn left')
                twist.linear.x = 0
                twist.angular.z = self.angular
                self.ros_ctrl.pub_vel.publish(twist)
                sleep(0.2)
            elif self.front_warning < 10 and self.Left_warning > 10 and self.Right_warning > 10:
                # print ('7、左右有障碍物，右转')
                # print ('7. There are obstacles on the left and right, turn right')
                twist.linear.x = 0
                twist.angular.z = -self.angular
                self.ros_ctrl.pub_vel.publish(twist)
                sleep(0.4)
            elif self.front_warning < 10 and self.Left_warning > 10 and self.Right_warning <= 10:
                # print ('8、左有障碍物，右转')
                # print ('8, there is an obstacle on the left, turn right')
                twist.linear.x = 0
                twist.angular.z = -self.angular
                self.ros_ctrl.pub_vel.publish(twist)
                sleep(0.2)
            elif self.front_warning < 10 and self.Left_warning <= 10 and self.Right_warning > 10:
                # print ('9、右有障碍物，左转')
                # print ('9, there is an obstacle on the right, turn left')
                twist.linear.x = 0
                twist.angular.z = self.angular
                self.ros_ctrl.pub_vel.publish(twist)
                sleep(0.2)
            elif self.front_warning <= 10 and self.Left_warning <= 10 and self.Right_warning <= 10:
                # print ('10、没有障碍物，前进')
                # print ('10, no obstacles, go forward')
                # EDIT STUFF HERE FOR NO AVOID MOVEMENT
                twist.linear.x = self.linear
                twist.angular.z = self.angular
                self.ros_ctrl.pub_vel.publish(twist)
                # If stuff in top does not work, then comment out the top part and uncomment the bottom 2 lines
                # spin = self.Spin(360)
                # if spin == True: self.Command = "finish"
            self.r.sleep()
            # else : self.ros_ctrl.pub_vel.publish(Twist())


if __name__ == '__main__':
    rospy.init_node('laser_Avoidance', anonymous=False)
    tracker = laserAvoid()
    tracker.robot_move()
    rospy.spin()
    tracker.cancel()