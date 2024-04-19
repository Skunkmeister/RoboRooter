
from ROS.RosHandler import RosHandler, process_topics
import threading


if __name__ == "__main__":
    rh = RosHandler()
    ros_thrd = threading.Thread(target=rh.subscribe_topics,args=())
    ros_thrd.start()

    try:
        while True:
            process_topics(rh)
            print(rh.rgb_img)
    except KeyboardInterrupt:
        print('Shutting down')
    finally:
        rh.client.terminate()