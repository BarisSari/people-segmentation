import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mrcnn.config
import mrcnn.visualize
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path


def segment_people(image, masks, class_ids):
    # Create a new solid-black image the same size as the original image
    masked = np.zeros(image.shape)

    # Loop over each detected object's mask
    for i in range(masks.shape[2]):
        # If the detected object isn't a person (class_id == 1), skip it
        if class_ids[i] != 1:
            continue

        # Draw the mask for the current object in white
        mask = masks[:, :, i]
        color = tuple(np.random.randint(256, size=3))

        masked = mrcnn.visualize.apply_mask(masked, mask, color, alpha=1.0)

    return masked.astype(np.uint8)


# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class


# Root directory of the project
ROOT_DIR = Path(".")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
weights = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(weights):
    mrcnn.utils.download_trained_weights(weights)

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Load pre-trained model
model.load_weights(weights, by_name=True)

for i in range(8, 5001):

    filename = str(i).zfill(4) + ".jpg"
    # Load the image we want to run detection on
    image_path = str(ROOT_DIR / "val2017" / filename)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_image = image[:, :, ::-1]

    # Run the image through the model
    results = model.detect([rgb_image], verbose=1)

    # Visualize results
    r = results[0]
    masked_image = segment_people(rgb_image, r['masks'], r['class_ids'])

    # Show the result on the screen
    # plt.imshow(masked_image.astype(np.uint8))
    # plt.show()
    cv2.imwrite("output/" + filename, gray+masked_image.astype(np.uint8))
