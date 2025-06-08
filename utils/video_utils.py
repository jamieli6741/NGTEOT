import cv2
import imutils
import numpy as np


def extract_video_frames(video_path, compress_rate=5, resize_width=500):
    # Load video and extract frames
    print(f"Loading video: {video_path}")
    vs = cv2.VideoCapture(video_path)

    if not vs.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Extract frames with compression
    print(f"Extracting frames (compression rate: {compress_rate})...")
    frame_list = []
    frame_cnt = 0

    while vs.isOpened():
        ret, frame = vs.read()
        if not ret:
            break

        if frame_cnt % compress_rate == 0:
            frame = imutils.resize(frame, width=resize_width)
            frame_list.append(frame)

        frame_cnt += 1

    frame_list = np.array(frame_list)
    vs.release()
    cv2.destroyAllWindows()
    print(f"Succesfully extracted {len(frame_list)} frames")
    return frame_list

def write_video(output_path, frames, fps=10):
    print("Writing Output Video...")
    height, width, _ = frames[0].shape
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for image in frames:
        video.write(image)
    video.release()
    print("Output Video Successfully Written to :{}".format(output_path))
