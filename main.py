from autostereogram.converter import StereogramConverter
from skimage import color
import matplotlib.image as mpimg
import numpy as np

source_image = mpimg.imread("bear.png")

# Use numpy to randomly generate some noise
image_data = np.array(source_image * 255, dtype=int)

converter = StereogramConverter()
result = converter.convert_depth_to_stereogram(image_data).astype(np.uint8)

mpimg.imsave("stereogram_result.png", result, cmap='gray')