import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mrcnn.config
import mrcnn.visualize as visualize
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import patches
import random
from matplotlib.patches import Polygon
import colorsys


# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class


class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Root directory of the project
ROOT_DIR = Path(".")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())
# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)


def get_ax(rows=1, cols=1, size=32):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, boxes, masks, class_ids, class_names, scores=None, colors=None):

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    f, ax = plt.subplots(1, figsize=(16, 16))

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]

        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=4,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        caption = "{} {:.3f}".format(label, score) if score else label

        ax.text(x1, y1-4, caption, color='w', size=13, backgroundcolor=color)

    ax.imshow(masked_image.astype(np.uint8))
    plt.show()
    # f.savefig('output3.jpg')

# used to test whether detection works or no
# for i in range(0, 1):
#     filename = str(i).zfill(4) + ".jpg"
#     # Load the image we want to run detection on
#     image_path = str(ROOT_DIR / "val2017" / filename)
#     image = cv2.imread(image_path)
#     # Convert the image from BGR color (which OpenCV uses) to RGB color
#     rgb_image = image[:, :, ::-1]
#
#     # Run the image through the model
#     results = model.detect([rgb_image], verbose=1)
#     # print(i)
#     # Visualize results
#     r = results[0]
#     mrcnn.visualize.display_instances(rgb_image, r['rois'], r['masks'], r['class_ids'],
#                                       class_names, r['scores'], show_mask=False, show_bbox=True)


image_path = str(ROOT_DIR / "val2017" / "0000.jpg")
image = cv2.imread(image_path)
rgb_image = image[:, :, ::-1]
results = model.detect([rgb_image], verbose=1)
r = results[0]
display_instances(rgb_image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
# cv2.imshow("test.jpg", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
