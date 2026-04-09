import cv2
import numpy as np
import math

def draw_line(img, pt1, pt2, color=(255,255,255)):
    cv2.line(img, (int(pt1[0]),int(pt1[1])), (int(pt2[0]),int(pt2[1])), color, 4)
    return img

def draw_pts(img, points):
    for pt in points:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 7, (0,255,0), -1)
        cv2.circle(img, (int(pt[0]), int(pt[1])), 10, (255,0,255), 2)
    return img

def draw_connection(img, points, connection):
    for k, (i,j) in enumerate(connection):
        color = (0,0,255) if k == len(connection) - 1 else (255,255,255)
        img = draw_line(img, points[i], points[j], color)
    return img

def calculatePushupAngle(img, pt1, pt2, pt3, color=(255, 255, 0), thickness=2, font_scale=0.6):
  pt1 = np.array(pt1) # shoulder keypoint
  pt2 = np.array(pt2) # elbow keypoint
  pt3 = np.array(pt3) # wrist keypoint

  v1 = pt1 - pt2  # shoulder-elbow vector
  v2 = pt3 - pt2  # wrist-elbow vector

  # Normalize magnitudes and compute angle
  unit_v1 = v1 / np.linalg.norm(v1)
  unit_v2 = v2 / np.linalg.norm(v2)
  dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
  angle_rad = np.arccos(dot_product)
  angle_deg = np.degrees(angle_rad)
  angle_deg = int(round(angle_deg))

  # Draw angle text
  cv2.putText(img, f'{angle_deg} degrees', (int(pt2[0]) + 13, int(pt2[1])), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

  return img, angle_deg

def rotate_point(pt, center, angle_deg):
    angle_rad = math.radians(angle_deg)
    x, y = pt
    cx, cy = center
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    # Translate to origin
    x_shifted, y_shifted = x - cx, y - cy

    # Rotate point
    x_rot = x_shifted * cos_a - y_shifted * sin_a
    y_rot = x_shifted * sin_a + y_shifted * cos_a

    return (x_rot + cx, y_rot + cy)

def align_points_to_fixed_line(
    points,
    pt1,
    pt2,
    frame,
    fixed_start,
    fixed_y=50,
    fixed_length=600,
    padding=15
):
    h, w, _ = frame.shape

    # Calculate line between shoulder and ankle
    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
    orig_len = math.hypot(dx, dy)
    center = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)

    # Calculate rotation angle to align line horizontally
    angle = -math.degrees(math.atan2(dy, dx))
    rot = [rotate_point(p, center, angle) for p in points]
    rot_pt1, rot_pt2 = rotate_point(pt1, center, angle), rotate_point(pt2, center, angle)
    rot_center = ((rot_pt1[0] + rot_pt2[0]) / 2, (rot_pt1[1] + rot_pt2[1]) / 2)

    # Scale line to arbitrary fixed_length
    scale = fixed_length / orig_len
    scaled = [(rot_center[0] + (x - rot_center[0]) * scale,
               rot_center[1] + (y - rot_center[1]) * scale) for x, y in rot]

    # Shift to fixed position
    shift_x, shift_y = (w/2 - rot_center[0]), (fixed_y - rot_center[1])
    shifted = [(x + shift_x, y + shift_y) for x, y in scaled]

    # Clamp points to stay within frame boundaries
    return [(max(padding, min(w - padding, x)),
             max(padding, min(h - padding, y))) for x, y in shifted]

def determine_pushup_direction_and_count(img, angle, counter, stage, align_pts):
    font, color = cv2.FONT_HERSHEY_COMPLEX, (255, 255, 255)
    scale = 2
    thick = 4
    pos = (20, 65)

    nose_y = align_pts[0][1] # y-postition nose
    elbow_y = align_pts[1][1] # y-position elbow

    if angle < 90 and elbow_y < nose_y and stage == 'up':
      stage = 'down'

    elif angle > 130 and stage == 'down':
      counter += 1
      stage = 'up'

    cv2.putText(img, str(counter), pos, font, scale, color, thick, cv2.LINE_AA)

    return img, counter, stage