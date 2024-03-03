# %%
import cv2
import torch
import mediapipe as mp
import numpy as np
from scipy.interpolate import RectBivariateSpline

# %%
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to("cuda")
midas.eval()

# %%
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
print(dir(transforms))
transform = transforms.dpt_transform


# %%
# Converting Depth to distance
def depth_to_distance(depth_value, depth_scale):
    return -1.0 / (depth_value * depth_scale)

# %%
import os
from time import time
from tqdm import tqdm


def convert_video_to_depth_frames(video_path: str, output_folder_path: str):
    os.makedirs(output_folder_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_counter = 0  # Initialize counter
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no frames left

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgbatch = transform(img).to("cuda")

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

        # Save the frame
        filename = os.path.join(output_folder_path, f"{frame_counter}.png")
        cv2.imwrite(filename, output_norm * 255)  # Multiply by 255 to scale to 0-255

        frame_counter += 1  # Increment the counter

        pbar.update(1)
        if cv2.waitKey(2) & 0xFF == ord("q"):
            break

    pbar.close()

    cap.release()
    cv2.destroyAllWindows()
