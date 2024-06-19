import cv2
from pupil_apriltags import Detector
from djitellopy import Tello
import ctypes

from task import Task1, Task2  # Importing the Task1 class

def keyboard(drone, key):
    print("key:", key)
    fb_speed = 40
    lf_speed = 40
    ud_speed = 50
    degree = 30

    if key == ord("1"):
        drone.takeoff()
    elif key == ord("2"):
        drone.land()
    elif key == ord("3"):
        drone.send_rc_control(0, 0, 0, 0)
        print("stop!!!!!")
    elif key == ord("w"):
        drone.send_rc_control(0, fb_speed, 0, 0)
        print("forward!!!!!")
    elif key == ord("s"):
        drone.send_rc_control(0, (-1) * fb_speed, 0, 0)
        print("backward!!!!!")
    elif key == ord("a"):
        drone.send_rc_control((-1) * lf_speed, 0, 0, 0)
        print("left!!!!!")
    elif key == ord("d"):
        drone.send_rc_control(lf_speed, 0, 0, 0)
        print("right!!!!!")
    elif key == ord("z"):
        drone.send_rc_control(0, 0, ud_speed, 0)
        print("down!!!!!")
    elif key == ord("x"):
        drone.send_rc_control(0, 0, (-1) * ud_speed, 0)
        print("up!!!!!")
    elif key == ord("c"):
        drone.send_rc_control(0, 0, 0, degree)
        print("rotate!!!!!")
    elif key == ord("v"):
        drone.send_rc_control(0, 0, 0, (-1) * degree)
        print("counter rotate!!!!!")
    elif key == ord("5"):
        height = drone.get_height()
        print(height)
    elif key == ord("6"):
        battery = drone.get_battery()
        print(battery)
    elif key == ord("7"):
        emergency = drone.emergency()
        print(emergency)

def main():

    drone = Tello()
    drone.connect()
    print(drone.get_battery())
    drone.streamon()
    drone.takeoff()
    drone.move_up(50)

    detector = Detector(families="tag36h11")
    #calibration = cv2.FileStorage("./test.xml", cv2.FileStorage_READ)
    
    camera_params = [313.34733040918235, 296.949736955647, 437.5629229985855, 421.367285388061]

    #task1 = Task1(drone)
    #task1_finished = False
    host =  'localhost'
    port = 8888
    task2 = Task2(drone, host, port)
    task2_finished = False

    while True:
        frame = drone.get_frame_read().frame
        tag_list = detector.detect(
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=0.105
        )

        #if not task1_finished:
        #    task1_finished = task1.run(tag_list, frame)
        #    if task1_finished:
        #        drone.send_rc_control(0, 0, 0, 0)
        if not task2_finished:
            task2_finished = task2.run(tag_list, frame)
            if task2_finished:
                drone.send_rc_control(0, 0, 0, 0)

        # draw corners
        for tag in tag_list:
            for corner in tag.corners:
                cv2.circle(frame, list(map(int, corner)), 5, (0, 0, 255), -1)

        cv2.imshow("drone", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(30)
        if key != -1:
            keyboard(drone, key)
        if (key & 0xFF) == ord("q") or task2_finished:
            break

    cv2.destroyAllWindows()
    drone.land()
    drone.streamoff()
    
    
if __name__ == "__main__":
    main()
