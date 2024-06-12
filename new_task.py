import time
import numpy as np
import math
import cv2
import argparse
import requests
from typing import List
from Tracker import Tracker  # Assuming this module contains the Tracker class and helper functions
from pid import Controller

from tellosrc.base import ResourceThread, StoppableThread
from tellosrc.receivers.detection import DetectionReceiver
from tellosrc.receivers.image import ImageReceiver
from tellosrc.receivers.state import StateReceiver

class Task1:
    def __init__(self, drone):
        self.drone = drone
        self.tracker = Tracker.Tracker()
        self.left_tag_last_seen = time.time()
        self.right_tag_last_seen = time.time()
        self.left_track_id = None
        self.right_track_id = None
        self.initialized = False
        self.controller = Controller(1.0, 0.0, 0.1)  # Initialize PID controller with some gains
        self.last_time = time.time()
        self.tolerance_x = 20
        self.tolerance_y = 20
        self.tolerance_yaw = 5  # Assuming 5 degrees as tolerance for yaw

    def initialize_tags(self, confirmed_tracks, frame_width):
        # Initialize left and right tracks based on initial positions
        for track in confirmed_tracks:
            x_min, y_min, x_max, y_max, tag_id, corner1_x, corner1_y, corner2_x, corner2_y, corner3_x, corner3_y, corner4_x, track_id = map(int, track)
            if tag_id == 0:
                centroid_x = (x_min + x_max) / 2
                if centroid_x < frame_width / 2:
                    if self.left_track_id is None:
                        self.left_track_id = track_id
                else:
                    if self.right_track_id is None:
                        self.right_track_id = track_id

        self.initialized = True

    def move_to_gate(self, gate_side):
        # Determine gate position based on side
        if gate_side == 'left':
            gate_tag_id = 1
        else:
            gate_tag_id = 2

        # Search for gate tag in the frame
        gate_tag = None
        for tag in self.tracker.tracks:
            if tag['tag_id'] == gate_tag_id:
                gate_tag = tag
                break

        if gate_tag is not None:
            x_min, y_min, x_max, y_max, _, _, _, _, _, _, _, _, _, _ = map(int, gate_tag['det'])
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
                self.drone.send_rc_control(0, 50, 0, 0)  # Move forward to pass through the gate


    def run(self, tag_list, frame):
        frame_size = frame.shape[:2]
        frame_width = frame_size[1]
        detections = []

        for tag in tag_list:
            # 0 is drones' tag, 1 is left gate, 2 is right gate
            if tag.tag_id in [0, 1, 2]:
                x_min, y_min = np.min(tag.corners, axis=0)
                x_max, y_max = np.max(tag.corners, axis=0)
                detections.append([x_min, y_min, x_max, y_max, tag.tag_id, tag.corners[0][0], tag.corners[0][1], tag.corners[1][0], tag.corners[1][1], tag.corners[2][0], tag.corners[2][1], tag.corners[3][0], tag.corners[3][1]])

        detections = np.array(detections)
        confirmed_tracks = self.tracker.update(detections, frame_size)

        if not self.initialized:
            self.initialize_tags(confirmed_tracks, frame_width)

        for track in confirmed_tracks:
            x_min, y_min, x_max, y_max, tag_id, corner1_x, corner1_y, corner2_x, corner2_y, corner3_x, corner3_y, corner4_x, track_id = map(int, track)
            if track_id == self.left_track_id:
                self.left_tag_last_seen = time.time()
            elif track_id == self.right_track_id:
                self.right_tag_last_seen = time.time()
                
            # Draw bounding box at tag 0, 1, 2
            cv2.putText(frame, f"ID: {track_id}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Determine which gate to go through
        current_time = time.time()
        if current_time - self.left_tag_last_seen > 2:  # If left tag not seen for 2 seconds
            print("Left tag lost, going through the right gate")
            self.move_to_gate('right')
        elif current_time - self.right_tag_last_seen > 2:  # If right tag not seen for 2 seconds
            print("Right tag lost, going through the left gate")
            self.move_to_gate('left')
        else:
            print("Both tags in sight or none, hovering in place")
            self.drone.send_rc_control(0, 0, 0, 0)  # Hover in place

        return False  # Returning False until task is finished
            
class Task2(StoppableThread):
    def __init__(self, drone, detector_url, tolerance_x=50, tolerance_y=50, tolerance_z=0.1):
        super().__init__()
        self.img_size = 416       # Adjust as needed
        self.conf_threshold = 0.5 # Adjust as needed
        self.nms_threshold = 0.45 # Adjust as needed
        
        self.state_receiver = StateReceiver()  # Receive state from Tello.
        self.image_receiver = ImageReceiver()  # Receive image from Tello.
        self.detection_receiver = DetectionReceiver(  # Detect objects in received image.
            self.image_receiver,
            self.img_size,
            self.conf_threshold,
            self.nms_threshold,
        )
        
        self.drone = drone
        self.detector_url = detector_url
        self.tolerance_x = tolerance_x
        self.tolerance_y = tolerance_y
        self.tolerance_z = tolerance_z
        self.tracker = Tracker()
        self.target_tag_id = None
        self.controller = Controller(1.0, 0.0, 0.1)  # Initialize PID controller with some gains
        self.last_time = time.time()

    def determine_target_tag(self, detection_class):
        class_to_tag_map = {
            1: 3,  # If detection class is 1, target AprilTag ID is 3
            2: 4,  # If detection class is 2, target AprilTag ID is 4
            3: 5,  # If detection class is 3, target AprilTag ID is 5
            4: 6   # If detection class is 4, target AprilTag ID is 6
        }
        self.target_tag_id = class_to_tag_map.get(detection_class, None)

    def get_pitch(self, R):
        sin_pitch = -R[2, 0]
        cos_pitch = np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)
        pitch = np.arctan2(sin_pitch, cos_pitch)
        return pitch

    def match_landing_target(self, frame, x, y, z, yaw):
        frame_center_x = frame.shape[1] / 2
        frame_center_y = frame.shape[0] / 2
        # Check if the drone is within the tolerance range of the target position and orientation
        return abs(x - self.x_target) < self.tolerance_x and \
               abs(y - self.y_target) < self.tolerance_y and \
               abs(z - self.z_target) < self.tolerance_z and \
               abs(yaw) < 5  # Assuming yaw tolerance is 5 degrees

    def extract_and_match_tag(self, tag_info: List[np.ndarray], target_tag_id: int):
        is_finish = False
        target_tag = None
        for tag in tag_info:
            if tag[4] == target_tag_id:
                target_tag = tag
                break
            
        if target_tag is not None:
            R = target_tag[11:20].reshape(3, 3)
            x, y, z = target_tag[7:10] * 100
            yaw = self.get_pitch(R)
            yaw = math.degrees(yaw)
            frame = self.drone.get_frame_read().frame
            
            
            if self.match_landing_target(frame, x, y, z, yaw):
                is_finish = True
                self.drone.send_rc_control(0, 0, 0, 0)
            else:
                current_time = time.time()
                dt = current_time - self.last_time
                self.last_time = current_time
                x_error = x - self.x_target
                y_error = y - self.y_target
                z_error = z - self.z_target
                yaw_error = yaw
                self.controller.update(self.drone, x_error, y_error, z_error, yaw_error, dt)
        else:
            self.drone.send_rc_control(0, 0, 0, 0)
        return is_finish

    def run(self, frame, tag_list):
        
        prev_id = None
        while not self.stopped():
            id, detection = self.detection_receiver.get_result()
            _, (state,) = self.state_receiver.get_result()
            if (id is None) or (id == prev_id):
                continue
            (_, bboxes, scores, labels, names) = detection
            print("-" * 80)
            print("Battery: %d%%" % state["bat"])
            print("X Speed: %.1f" % state["vgx"])
            print("Y Speed: %.1f" % state["vgy"])
            print("Z Speed: %.1f" % state["vgz"])
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
