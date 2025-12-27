import cv2
import torch
import torch.nn as nn
import numpy as np
import os

from module_def import EmotionDetector

# main function        
if __name__ == '__main__':
    # load file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cascade_file = os.path.join(script_dir, "haarcascade_frontalface_default.xml")
    model_file = os.path.join(script_dir, "face_expression_excel.pth")
    video_file = os.path.join(script_dir, "demo.mp4")
    output_file = os.path.join(script_dir, "video_result.avi")

    # initialize detector
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    try:
        # check if files exist
        for f in [cascade_file, model_file, video_file]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Required file not found: {f}")
    
        detector = EmotionDetector(cascade_path=cascade_file, model_path=model_file, device=device)

        # prepare to read and write video

        # open video file for reading
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_file}")

        # retrieve video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # create VideoWriter object to save output video
        fourcc = cv2.VideoWriter.fourcc(*'XVID')
        out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
        
        print(f"Processing video: {video_file}...")
        print(f"Frame rate: {fps}, Size: {frame_width}x{frame_height}")

        # Start a loop and process the video frame by frame until the video ends
        while cap.isOpened():
            # read one frame from the video
            ret, frame = cap.read()
            # if ret is False, then video has ended
            if not ret:
                break

            # porcess the current frame with the detector and get annotated frame and predictions
            annotated_frame, predictions = detector.detect_and_predict(frame)
            # write the processed and annotated frame to the "recorder"
            out.write(annotated_frame)

        # release video objects and close all OpenCV windows
        print("Processing finished.")
        cap.release()  
        out.release()  # shutdown the "recorder", finalizing the video file
        
        print(f"Result video saved to: {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()