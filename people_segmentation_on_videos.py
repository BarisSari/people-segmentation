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
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    masked = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Loop over each detected object's mask
    for i in range(masks.shape[2]):
        # If the detected object isn't a person (class_id == 1), skip it
        if class_ids[i] != 1:
            continue

        # Draw the mask for the current object in white
        mask = masks[:, :, i]
        color = tuple(np.random.randint(256, size=3))
        # color = mrcnn.visualize.random_colors(3)
        masked = mrcnn.visualize.apply_mask(masked, mask, color=color, alpha=1.0)

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

input_video = 'demo/1.mp4'
capture = cv2.VideoCapture(input_video)
fps = 25.0
width = int(capture.get(3))
height = int(capture.get(4))
fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
out = cv2.VideoWriter("new_video.avi", fcc, fps, (width, height))

while True:
    ret, frame = capture.read()
    results = model.detect([frame], verbose=0)
    r = results[0]
    masked_frame = segment_people(frame, r['masks'], r['class_ids'])
    cv2.imshow('video', masked_frame)

    # Recording Video
    out.write(masked_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
