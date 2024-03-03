import numpy as np
import matplotlib.image as mpimg
from autostereogram.converter import StereogramConverter
import skvideo.io
from skimage import color
import os
import time

def main():
    image_folder = "saved_frames_real_round3"
    image_files = os.listdir(image_folder)
    frame_converter = StereogramConverter()

    # Generate a more uniform texture pattern
    height, width = 128 * 8, 128 * 8
    texture_pattern = np.random.randint(0, 256, size=(height, width), dtype=int)

    # Create a vertical stripe pattern
    #stripe_width = 16  # Width of each stripe
    #for i in range(0, width, stripe_width * 2):
     #   texture_pattern[:, i:i + stripe_width] = 64  # Dark stripes on a black background

    with skvideo.io.FFmpegWriter("outputvideofinal2texture.mp4") as video_writer:
        for i in range(len(image_files)):
            start_time = time.time()
            image_path = os.path.join(image_folder, f"frame{i}.png")
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
            print(f"Frame {i} processed in {time.time() - start_time} seconds")

if __name__ == "__main__":
    main()
