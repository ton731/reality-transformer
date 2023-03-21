import cv2
from pathlib import Path
import os
from tqdm import tqdm




# compare two image frames and compute the similarity
MSE_SIMILARITY_THRESHOLD = 100
def image_is_similar(img1, img2) -> bool:
    # Convert the images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute the MSE between the two images
    mse = ((img1_gray - img2_gray) ** 2).mean()

    if mse > MSE_SIMILARITY_THRESHOLD:
        return False
    
    return True




# check an image every 3 seconds
CHECK_IMG_SECONDS = 1

def read_video_to_frames(video_path, img_folder, video_index):
    # Open the video file
    video = cv2.VideoCapture(str(video_path))

    # stats
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = round(video.get(cv2.CAP_PROP_FPS))
    print("Getting frames from:", video_path)
    print("length:", length)
    print("fps:", fps)

    # Initialize frame counter and loop through video frames
    count = 0
    last_frame = None
    while True:
        print(f"frame {count}/{length}")

        # Read a frame from the video
        ret, frame = video.read()

        # If reach the end of the video, break out of the loop
        if not ret:
            break

        # Write the current frame as a JPEG image file per several second
        if count % (fps * CHECK_IMG_SECONDS) == 0:
            if last_frame is None or not image_is_similar(last_frame, frame):
                cv2.imwrite(str(img_folder / f"{video_index}_{count}.jpg"), frame)
                last_frame = frame

        # Increment the frame counter
        count += 1

    # Release the video file
    video.release()




def main():
    data_folder = Path("data")
    data_folder.mkdir(parents=True, exist_ok=True)

    reality_video_folder = data_folder / "reality_videos"
    animation_video_folder = data_folder / "animation_videos"
    reality_image_folder = data_folder / "reality_images"
    animation_image_folder = data_folder / "animation_images"

    reality_image_folder.mkdir(parents=True, exist_ok=True)
    animation_image_folder.mkdir(parents=True, exist_ok=True)


    reality_vedio_names = []
    animation_vedio_names = ["Summer Wars.mkv", "The Garden of Words.mp4"]


    # for i, video_name in enumerate(os.listdir(reality_video_folder)):
    for i, video_name in enumerate(reality_vedio_names):
        video_path = reality_video_folder / video_name
        read_video_to_frames(video_path, reality_image_folder, i)

    # for i, video_name in enumerate(os.listdir(animation_video_folder)):
    for i, video_name in enumerate(animation_vedio_names):
        video_path = animation_video_folder / video_name
        read_video_to_frames(video_path, animation_image_folder, i)


    
if __name__ == "__main__":
    main()

    






