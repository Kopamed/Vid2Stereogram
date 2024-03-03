import numpy as np
import matplotlib.image as mpimg
from autostereogram.converter import StereogramConverter
import skvideo.io
from skimage import color
import os
import time
from tqdm import tqdm

def convert_depth_frames_to_video(frames_folder="input/example_frames", output_video="output/outputvideo.mp4"):
    image_files = os.listdir(frames_folder)
    frame_converter = StereogramConverter()

    # Generate a more uniform texture pattern
    height, width = 128 * 8, 128 * 8
    texture_pattern = np.random.randint(0, 256, size=(height, width), dtype=int)

    # Create a vertical stripe pattern
    #stripe_width = 16  # Width of each stripe
    #for i in range(0, width, stripe_width * 2):
     #   texture_pattern[:, i:i + stripe_width] = 64  # Dark stripes on a black background

    with skvideo.io.FFmpegWriter(output_frame) as video_writer:
        pbar = tqdm(total=len(image_files), unit="frame")

        for i in range(len(image_files)):
            image_path = os.path.join(frames_folder, f"{i}.png")
            image_data = np.array(mpimg.imread(image_path) * 255, dtype=int)
            image_data = 255 - image_data

            # Convert depth map to stereogram using the generated uniform texture pattern for each frame
            image_data = frame_converter.convert_depth_to_stereogram_with_texture(
                depth_map=image_data,
                texture=texture_pattern,
            )
            output_frame = image_data.astype(np.uint8)
            output_frame = color.gray2rgb(output_frame)
            video_writer.writeFrame(output_frame)
            pbar.update(1)
        
        pbar.close()

if __name__ == "__main__":
    convert_depth_frames_to_video()
