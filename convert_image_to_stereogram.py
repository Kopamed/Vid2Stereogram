import argparse
import os
from video_to_depth_frames import convert_video_to_depth_frames
from frames_to_stereogram import convert_depth_frames_to_video

import cv2
import torch
import mediapipe as mp
import numpy as np
from scipy.interpolate import RectBivariateSpline
from autostereogram.converter import StereogramConverter
from skimage import color
import matplotlib.image as mpimg


# Converting Depth to distance
def depth_to_distance(depth_value, depth_scale):
    return -1.0 / (depth_value * depth_scale)


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Convert images to spectograms.")

    # Add a positional argument
    parser.add_argument("input_path", type=str, help="Path to the input image file.")
    parser.add_argument("output_path", type=str, help="Path to the output image file.")

    # Parse the arguments
    args = parser.parse_args()

    temp_depth_image = "temp_depth_image.png"

    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to("cpu")
    midas.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    print(dir(transforms))
    transform = transforms.dpt_transform

    img = cv2.imread(args.input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocess the image and prepare it for the model
    imgbatch = transform(img).to("cpu")

    # Making a prediction
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    # Normalizing the output predictions for cv2 to read.
    output_norm = cv2.normalize(
        output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    # Save the depth image
    cv2.imwrite(temp_depth_image, output_norm * 255)  # Multiply by 255 to scale to 0-255

    source_image = mpimg.imread(temp_depth_image)

    # Use numpy to randomly generate some noise
    image_data = np.array(source_image * 255, dtype=int)
    image_data = 255 - image_data

    converter = StereogramConverter()
    result = converter.convert_depth_to_stereogram(image_data).astype(np.uint8)

    # Save the final output
    mpimg.imsave(args.output_path, result, cmap="gray")

    os.remove(temp_depth_image)
    print("Done!")


if __name__ == "__main__":
    main()
