import cv2
import os

def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, frame = cap.read()
    
    while success:
        cv2.imwrite(os.path.join(output_folder, f"frame_{frame_count:05d}.png"), frame)
        success, frame = cap.read()
        frame_count += 1
    
    cap.release()

extract_frames('movie/ikiru1952.mp4', 'frames')
