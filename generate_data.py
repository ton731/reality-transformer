import cv2
from pathlib import Path
import os


def read_video_to_frames(video_path, img_folder, video_index):
    print("Getting frames from:", video_path)
    # Open the video file
    video = cv2.VideoCapture(str(video_path))

    # stats
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = round(video.get(cv2.CAP_PROP_FPS))
    print("length:", length)
    print("fps:", fps)

    # Initialize frame counter and loop through video frames
    count = 0
    while True:
        # Read a frame from the video
        ret, frame = video.read()

        # If reach the end of the video, break out of the loop
        if not ret:
            break

        # Write the current frame as a JPEG image file per second
        if count % fps == 0:
            cv2.imwrite(str(img_folder / f"{video_index}_{count}.jpg"), frame)

        # Increment the frame counter
        count += 1

    # Release the video file
    video.release()




def main():
    data_folder = Path("data")
    reality_video_folder = data_folder / "reality_videos"
    animation_video_folder = data_folder / "animation_videos"
    reality_image_folder = data_folder / "reality_images"
    animation_image_folder = data_folder / "animation_images"

    for i, video_name in enumerate(os.listdir(reality_video_folder)):
        video_path = reality_video_folder / video_name
        read_video_to_frames(video_path, reality_image_folder, i)

    for i, video_name in enumerate(os.listdir(animation_video_folder)):
        video_path = animation_video_folder / video_name
        read_video_to_frames(video_path, animation_image_folder, i)


    
if __name__ == "__main__":
    main()

    






