# yolo-pushup-counter

## Overview
This project is a form feedback system using YOLO v8 pose estimation and OpenCV to count pushups in real-time from a video mp4 file. It captures a joint angle keypoint model between the shoulder-elbow-wrist to track the movement of the body to determine when a pushup is completed.  The pushup count is displayed in the top left corner of the video frame.

![Pushup_Counter_Trim-ezgif com-optimize](https://github.com/user-attachments/assets/7ae5985c-829e-441e-b0f2-63c0246485d5)

## Steps to Run
1. Create conda env from environment.yml: `conda env create -f environment.yml`
2. Activate conda ev: `conda activate yolo-pushup-counter`
3. Run the file: `python pushup_counter.py`

Hint: Uncomment out `cv2.imshow("Pushup Counter", frame)` in `pushup_counter.py` to see the video frames

## Future Enhancements
- Logic currently only works for a left side view of the person performing the pushup.  Next steps would be to add support for right side, front, and back views
- Add support for real-time video input from Raspberry Pi camera
