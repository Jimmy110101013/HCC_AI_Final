import time
import numpy as np
import math
import cv2
import argparse
import requests
from typing import List
from Tracker import Tracker  # Assuming this module contains the Tracker class and helper functions
from pid import Controller
from pupil_apriltags import Detector

from tellosrc.base import ResourceThread, StoppableThread
from tellosrc.receivers.detection import DetectionReceiver
from tellosrc.receivers.image import ImageReceiver
from tellosrc.receivers.state import StateReceiver

def estimate_camera_pose(frame):
    detector = Detector(families='tag36h11')
    fx = 313.34733040918235
    fy = 296.949736955647
    cx = 437.5629229985855
    cy = 421.367285388061
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    dist_coeffs = np.array([
    0.40408531, -1.78957082, 0.02626639, 0.01158871, 4.15482281
    ])
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray_frame)
    if not tags:
        return None, None

    tag = tags[0]
    corners = tag.corners

    # Define the 3D coordinates of the tag corners in the tag's coordinate system
    tag_size = 0.1  # meters
    object_points = np.array([
        [-tag_size / 2, -tag_size / 2, 0],
        [tag_size / 2, -tag_size / 2, 0],
        [tag_size / 2, tag_size / 2, 0],
        [-tag_size / 2, tag_size / 2, 0]
    ])

    # The corresponding 2D points in the image
    image_points = np.array(corners)

    # Solve for the camera pose
    success, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    if not success:
        return None, None

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    return translation_vector, rotation_matrix

class Task1:
    def __init__(self, drone):
        self.drone = drone
        self.tracker = Tracker()
        self.left_tag_seen_count = 0
        self.right_tag_seen_count = 0
        self.LR_diff = 0
        self.left_track_id = None
        self.right_track_id = None
        self.initialized = False
        self.checkpoint1_finished = False
        self.controller = Controller(1.0, 0.0, 0.1)  # Initialize PID controller with some gains
        self.last_time = time.time()
        self.tolerance_x = 20
        self.tolerance_y = 20
        self.tolerance_yaw = 5  # Assuming 5 degrees as tolerance for yaw

    def initialize_tags(self, confirmed_tracks, frame_width):
        # Initialize left and right tracks based on initial positions
        for track in confirmed_tracks:
            x_min, y_min, x_max, y_max, tag_id, corner1_x, corner1_y, corner2_x, corner2_y, corner3_x, corner3_y, corner4_x, corner4_y, track_id = track
            x_min, y_min, x_max, y_max, tag_id = map(int, [x_min, y_min, x_max, y_max, tag_id])
            if tag_id == 0:
                centroid_x = (x_min + x_max) / 2
                if centroid_x < frame_width / 2:
                    if self.left_track_id is None:
                        print(f"left_track_id is set to {track_id}")
                        self.left_track_id = track_id
                else:
                    if self.right_track_id is None:
                        print(f"right_track_id is set to {track_id}")
                        self.right_track_id = track_id

        self.initialized = True

    def find_gate(self, gate_side, gate_tag_id):
        try:
            # Search for gate tag in the frame
            gate_tag_track = None
            for track in self.tracker.tracks:
                if track['det'].tag_id == gate_tag_id:
                    gate_tag_track = track

            if gate_tag_track is not None:
                x_min, y_min, x_max, y_max, _, _, _, _, _, _, _, _, _, _ = map(int, gate_tag_track)
                gate_center_x = (x_min + x_max) / 2
                gate_center_y = (y_min + y_max) / 2
                frame = self.drone.get_frame_read().frame
                frame_center_x = frame.shape[1] / 2
                frame_center_y = frame.shape[0] / 2

                # Calculate errors
                x_error = gate_center_x - frame_center_x
                y_error = gate_center_y - frame_center_y
                z_error = self.z_target  # Assuming constant z_target for now
                yaw_error = 0  # Assuming no yaw error for now

                # Update the drone position
                current_time = time.time()
                dt = current_time - self.last_time
                self.last_time = current_time
                self.controller.update(self.drone, x_error, y_error, z_error, yaw_error, dt)

                # Check if the drone is aligned with the gate
                if abs(x_error) < self.tolerance_x and abs(y_error) < self.tolerance_y and abs(yaw_error) < self.tolerance_yaw:
                    print(f"Aligned with {gate_side} gate, moving forward")
                    
        except Exception as e:
            print(f"Error in find_gate: {e}. HaHa no gate detect but still move forward :)))")
                
    def move_to_gate(self, gate_side):
        # Determine gate position based on side
        if gate_side == 'left':
            gate_tag_id = 1
            print("move_to_gate send command -- left 30")
            self.drone.move_left(30)  # Move left by 30
            time.sleep(3)  # Adjust the sleep time as needed to achieve the desired movement
            print("move_to_gate send command -- forward 30")
            self.drone.move_forward(20)  # Move forward by 20
            print("move_to_gate send command -- wait PID")
            self.find_gate(gate_side, gate_tag_id)
            time.sleep(1)
            self.drone.send_rc_control(0,0,0,0)
            print("move_to_gate send command -- forward 40")
            self.drone.move_forward(40)  # Move forward by 40
        else:
            gate_tag_id = 2
            print("move_to_gate send command -- right 30")
            self.drone.move_right(30)  # Move left by 30
            time.sleep(3)  # Adjust the sleep time as needed to achieve the desired movement
            print("move_to_gate send command -- forward 30")
            self.drone.move_forward(20)
            print("move_to_gate send command -- wait PID")
            self.find_gate(gate_side, gate_tag_id)
            time.sleep(1)
            self.drone.send_rc_control(0,0,0,0)
            print("move_to_gate send command -- forward 40")
            self.drone.move_forward(40)  # Move forward by 80

    def run(self, tag_list, frame):
        self.tag_list = tag_list
        frame_size = frame.shape[:2]
        frame_width = frame_size[1]
        detections = []

        for tag in tag_list:
            # 0 is drones' tag, 1 is left gate, 2 is right gate
            if tag.tag_id in [0, 1, 2]:
                x_min, y_min = np.min(tag.corners, axis=0)
                x_max, y_max = np.max(tag.corners, axis=0)
                detections.append([x_min, y_min, x_max, y_max, tag.tag_id, 
                                   tag.corners[0][0], tag.corners[0][1], tag.corners[1][0], tag.corners[1][1], 
                                   tag.corners[2][0], tag.corners[2][1], tag.corners[3][0], tag.corners[3][1]])

        detections = np.array(detections)
        confirmed_tracks = self.tracker.update(detections, frame_size, checkpoint1_finished=self.checkpoint1_finished)

        if not self.initialized:
            self.initialize_tags(confirmed_tracks, frame_width)

        for track in confirmed_tracks:
            x_min, y_min, x_max, y_max, tag_id, _, _, _, _, _, _, _, _, track_id = track
            x_min, y_min, x_max, y_max, tag_id = map(int, [x_min, y_min, x_max, y_max, tag_id])
            if track_id == self.left_track_id:
                self.left_tag_seen_count += 1
                print(f"------------- left track seen count = {self.left_tag_seen_count} -------------")
            elif track_id == self.right_track_id:
                self.right_tag_seen_count += 1 
                print(f"------------- right track seen count = {self.right_tag_seen_count} -------------")
                
            # Draw bounding box at tag 0, 1, 2
            cv2.putText(frame, f"track ID: {track_id}", (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"tag ID: {tag_id}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Determine which gate to go through
        self.LR_diff = self.left_tag_seen_count - self.right_tag_seen_count
        print(f"  L-R diff = {self.LR_diff} times")
        if self.LR_diff < -10:  # If left tag not seen for 10 time
            print("Left tag lost, going through the right gate")
            self.checkpoint1_finished = True
            self.move_to_gate('right')
        elif self.LR_diff > 10:  # If right tag not seen for 10 time
            print("Right tag lost, going through the left gate")
            self.checkpoint1_finished = True
            self.move_to_gate('left')
        else:
            print("Both tags in sight or none, hovering in place")
            self.drone.send_rc_control(0, 0, 0, 0)  # Hover in place

        return False  # Returning False until task1 is finished
            
class Task2(StoppableThread):
    def __init__(self, drone, host, port):
        super().__init__()
        self.img_size = 640       # Adjust as needed
        self.conf_threshold = 0.1 # Adjust as needed
        self.nms_threshold = 0.45 # Adjust as needed
        self.host = host
        self.port = port
        
        self.image_receiver = ImageReceiver()  # Receive image from Tello.
        self.detection_receiver = DetectionReceiver(  # Detect objects in received image.
            self.image_receiver,
            self.img_size,
            self.conf_threshold,
            self.nms_threshold,
            url=f"http://{self.host}:{self.port}/api/yolov7"
        )
        
        self.drone = drone
        self.task2_finished = False
        self.detector_url = None
        self.tracker = Tracker()
        self.target_tag_id = None
        self.controller = Controller(1.0, 0.0, 0.1)  # Initialize PID controller with some gains
        self.last_time = time.time()

    def determine_target_tag(self, detection_class):
        class_to_tag_map = {
            'Gryffindor': 3,  # If detection class is red, target AprilTag ID is 3
            'Hufflepuff': 4,  # If detection class is yellow, target AprilTag ID is 4
            'Ravenclaw' : 5,  # If detection class is green, target AprilTag ID is 5
            'Slytherin' : 6   # If detection class is blue, target AprilTag ID is 6
        }
        self.target_tag_id = class_to_tag_map.get(detection_class, None)

    def get_pitch(self, R):
        sin_pitch = -R[2, 0]
        cos_pitch = np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)
        pitch = np.arctan2(sin_pitch, cos_pitch)
        return pitch

    def extract_and_match_tag(self, confirmed_tracks: List[np.ndarray], target_tag_id: int):
        is_finish = False
        target_tag = None
        for track in confirmed_tracks:
            if track['det'][4] == target_tag_id:
                target_tag = track['det'][4]
            
        if target_tag is not None:
            R = target_tag[11:20].reshape(3, 3)
            x, y, z = target_tag[7:10] * 100
            yaw = self.get_pitch(R)
            yaw = math.degrees(yaw)

            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time
            self.controller.update(self.drone, x, y, z, yaw, dt)
        else:
            self.drone.send_rc_control(0, 0, 0, 0)
        return is_finish

    def run(self, frame, tag_list):
        
        threads = [
            self.image_receiver,
            self.detection_receiver,
        ]
        for thread in threads:
            thread.start()
        
        prev_id = None
        
        while not self.task2_finished:
            id, obj_detection = self.detection_receiver.get_result()
            if (id is None) or (id == prev_id):
                continue
            (_, bboxes, scores, labels, names) = obj_detection

            for bbox, score, label, name in zip(bboxes, scores, labels, names):
                # center (x, y) and box size (w, h)
                x, y, w, h = bbox
                print(
                    ", ".join(
                        [
                            "Label: %d" % int(label),
                            "Name: %s" % name,
                            "Conf: %.5f" % score,
                            "center: (%.1f, %.1f)" % (x, y),
                            "size: (%.1f, %.1f)" % (w, h),
                        ]
                    )
                )
                
                self.determine_target_tag(label)
                
            prev_id = id
            
            if self.target_tag_id is None:
                print("No target tag ID determined.")
                return False
            
            elif self.target_tag_id == 3 :
                print("move_to_gate send command -- left 40")
                self.drone.move_left(40)  # Move left by 40
                time.sleep(1)
                print("move_to_gate send command -- forward 30")
                self.drone.move_forward(30)  # Move forward by 30
                self.task2_finished = True
                return True
            
            elif self.target_tag_id == 4 :
                print("move_to_gate send command -- left 20")
                self.drone.move_left(20)  # Move left by 20
                time.sleep(1)
                print("move_to_gate send command -- forward 30")
                self.drone.move_forward(30)  # Move forward by 30
                self.task2_finished = True
                return True
            elif self.target_tag_id == 5 :
                print("move_to_gate send command -- right 20")
                self.drone.move_right(20)  # Move right by 20
                time.sleep(1)
                print("move_to_gate send command -- forward 30")
                self.drone.move_forward(30)  # Move forward by 30
                self.task2_finished = True
                return True
            elif self.target_tag_id == 6 :
                print("move_to_gate send command -- right 40")
                self.drone.move_right(40)  # Move right by 40
                time.sleep(1)
                print("move_to_gate send command -- forward 30")
                self.drone.move_forward(30)  # Move forward by 30
                self.task2_finished = True
                return True
            
                
        '''
            frame = self.drone.get_frame_read().frame
            frame_size = frame.shape[:2]
            detections = []

            for tag in tag_list:
                if tag.tag_id == self.target_tag_id:
                    x_min, y_min = np.min(tag.corners, axis=0)
                    x_max, y_max = np.max(tag.corners, axis=0)
                    detections.append([x_min, y_min, x_max, y_max, tag.tag_id, tag.corners[0][0], tag.corners[0][1], tag.corners[1][0], tag.corners[1][1], tag.corners[2][0], tag.corners[2][1], tag.corners[3][0], tag.corners[3][1]])

            detections = np.array(detections)
            confirmed_tracks = self.tracker.update(detections, frame_size)

            is_finish = self.extract_and_match_tag(confirmed_tracks, self.target_tag_id)
            return is_finish
        '''
