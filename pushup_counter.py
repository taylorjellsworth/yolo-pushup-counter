import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from ultralytics import YOLO
from utils import align_points_to_fixed_line, draw_pts, draw_connection, calculatePushupAngle, determine_pushup_direction_and_count

def process_yolo_pushup(frame, p, side_indices, relative_indices, connection_body, connection_relative,
                 anchor_idx, counter, stage, fixed_start, fixed_length, pt1, pt2):
    side_points = [p[i] for i in side_indices]
    relative_points = [p[i] for i in relative_indices]

    frame = draw_pts(frame, side_points)
    frame = draw_connection(frame, side_points, connection_body)

    # Calculate joint angle (shoulder-elbow-wrist)
    frame, angle_deg = calculatePushupAngle(frame, p[anchor_idx], p[anchor_idx + 2], p[anchor_idx + 4])

    # Align body to fixed reference line (normalization)
    aligned_pts = align_points_to_fixed_line(
        relative_points, pt1, pt2, frame,
        fixed_start, fixed_length=fixed_length
    )

    frame = draw_pts(frame, aligned_pts[2:])
    frame = draw_connection(frame, aligned_pts, connection_relative)

    # calculate pushup direction and count
    frame, counter, stage = determine_pushup_direction_and_count(frame, angle_deg, counter, stage, aligned_pts)
    return frame, counter, stage


def execute_yolo_pushup_counter(path, model):
  pushup_count = 0
  pushup_stage = 'up'
  video_capture = cv2.VideoCapture(path)
  
  if not video_capture.isOpened():
      print("Opening Video Failed")
      return

  skeleton_connection_body = [(1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (1, 6)]
  skeleton_connection_relative = [(2, 3), (3, 4), (4, 5), (2, 5)]

  while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
      break

    frame = cv2.resize(frame, (1280, 720))
    results = model.predict(frame, conf=0.5, verbose=False)
    keypoints = results[0].keypoints.xy[0]
    n_array = keypoints.cpu().numpy()

    
    pt1 = n_array[5] # Left Shoulder Keypoint
    pt2 = n_array[15] # Left Ankle Keypoint

    frame, pushup_count, pushup_stage = process_yolo_pushup(
        frame,
        n_array,
        side_indices=[0, 5, 7, 9, 11, 13, 15],
        relative_indices=[0, 7, 5, 11, 13, 15],
        connection_body=skeleton_connection_body,
        connection_relative=skeleton_connection_relative,
        anchor_idx=5,
        counter=pushup_count,
        stage=pushup_stage,
        fixed_start=(690, 50),
        fixed_length=400,
        pt1=pt1,
        pt2=pt2
    )

    # Uncomment to show video frames
    # cv2.imshow("Pushup Counter", frame)

    # Exit on ESC key
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
      pushup_count = 0
      break

  return pushup_count


if __name__ == "__main__":
  model = YOLO("yolov8n-pose.pt")
  video_path = "./test_videos/test_pushup.mp4"

  pushup_count = execute_yolo_pushup_counter(video_path, model)
  print(f"Final pushup count: {pushup_count}")

