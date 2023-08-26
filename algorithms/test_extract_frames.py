import cv2

def extract_frames(video_path, num_frames=5):
    # Open the video file.
    cap = cv2.VideoCapture(video_path)
    
    # Get the total number of frames in the video.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the interval between frames we want to extract.
    interval = total_frames // (num_frames + 1)
    
    frames = []

    for i in range(1, num_frames + 1):
        # Set the position of which frame to capture.
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        
        # Read the frame.
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    # Release the video capture object.
    cap.release()

    return frames

video_path = "/data/xcao/code/uni_recognize_demo/test_misc/test_videos/dog_playing.mp4"
frames = extract_frames(video_path)

# If you want to save or display the frames:
for idx, frame in enumerate(frames):
    cv2.imwrite('/data/xcao/code/uni_recognize_demo/test_misc/out_frames/' + f'frame_{idx}.png', frame)
    # Alternatively, to display the frame:
    # cv2.imshow(f'frame_{idx}', frame)
    # cv2.waitKey(0)

# To close all OpenCV windows (if displaying):
# cv2.destroyAllWindows()
