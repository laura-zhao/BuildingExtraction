
# The MaskRCNN model library are implemented by Waleed Abdulla(https://github.com/waleedka)
# Need to install the MaskRCNN library first to run this script
import os
import sys
import random
import numpy as np
from PIL import Image, ImageDraw
import json
import geoio
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

def initialize_directories(root_dir_path):
    """
    Initialize and return key directories for Mask R-CNN based on the given root directory.

    Parameters:
    root_dir_path (str): The path to the root directory of the project.

    Returns:
    tuple: Contains paths for the root directory, dataset directory, model directory, and COCO model path.
    """
    root_dir = os.path.abspath(root_dir_path)
    dataset_dir = os.path.join(root_dir, "SpaceNet/Train/AOI_2_Vegas_Train")
    model_dir = os.path.join(root_dir, "logs")
    coco_model_path = os.path.join(root_dir, "mask_rcnn_coco.h5")
  
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)
      
    # Append the root directory to the system path
    sys.path.append(root_dir)

    return root_dir, dataset_dir, model_dir, coco_model_path


class SpaceNetConfig(Config):
    """
    Configuration for training on the SpaceNet dataset.
    Overrides values specific to the SpaceNet dataset.
    """
    NAME = "SpaceNet"
    BACKBONE = "resnet50"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2  # background + 1 building
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    RPN_ANCHOR_RATIOS = [0.25, 1, 4]
    TRAIN_ROIS_PER_IMAGE = 32
    USE_MINI_MASK = True
    STEPS_PER_EPOCH = 500
    VALIDATION_STEPS = 50
    MAX_GT_INSTANCES = 250
    DETECTION_MAX_INSTANCES = 350

def get_ax(rows=1, cols=1, size=8):
    """
    Return a Matplotlib Axes array to be used in all visualizations.
    Args:
        rows (int): Number of rows in the subplot.
        cols (int): Number of columns in the subplot.
        size (int): Size of the subplot.
    Returns:
        Matplotlib Axes array.
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

def fill_between(polygon):
    """
    Create a mask for a given polygon.
    Args:
        polygon (list of tuples): Coordinates of the polygon vertices.
    Returns:
        np.array: A boolean array representing the mask.
    """
    img = Image.new('1', (650, 650), False)
    ImageDraw.Draw(img).polygon(polygon, outline=True, fill=True)
    return np.array(img)

class SpaceNetDataset(utils.Dataset):
    """
    Generates the SpaceNet dataset.
    The dataset consists of building images and their annotations.
    """

    def load_dataset(self, dataset_dir, start=1, end=400):
        """
        Generate the requested number of images and their annotations.
        Args:
            dataset_dir (str): Directory containing the dataset.
            start (int): Starting index of images to load.
            end (int): Ending index of images to load.
        """
        self.add_class("SpaceNetDataset", 1, "building")
        images_dir = os.path.join(dataset_dir, "RGB-PanSharpen1/")
        annotations_dir = os.path.join(dataset_dir, "geojson/buildings/")

        for filename in os.listdir(images_dir)[start:end]:
            image_id = filename[31:-4]
            image_path = os.path.join(images_dir, filename)
            annotation_path = os.path.join(annotations_dir, f"buildings_AOI_2_Vegas_imgg{image_id}.geojson")
            self.add_image('SpaceNetDataset', image_id=image_id, path=image_path, annotation=annotation_path)

    def load_image(self, image_id):
        """
        Load an image from the dataset.
        Args:
            image_id (int): The ID of the image to load.
        Returns:
            np.array: The loaded image as a NumPy array.
        """
        image_info = self.image_info[image_id]
        path = image_info['path']
        image = Image.open(path)
        return np.array(image)

    def image_reference(self, image_id):
        """
        Return the path of the image for reference.
        Args:
            image_id (int): The ID of the image.
        Returns:
            str: The path of the image.
        """
        info = self.image_info[image_id]
        if info["source"] == "SpaceNetDataset":
            return info["path"]
        else:
            super().image_reference(image_id)

    def load_mask(self, image_id):
        """
        Generate instance masks for buildings of the given image ID.
        Args:
            image_id (int): The ID of the image for which to load masks.
        Returns:
            tuple: A tuple containing:
                - np.array: A boolean array of shape [height, width, instance count] with one mask per instance.
                - np.array: An array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotation_path = image_info['annotation']
        masks = []
        class_ids = []

        with open(annotation_path) as f:
            data = json.load(f)
            allBuildings = data['features']

            for building in allBuildings:
                coordinates = building['geometry']['coordinates'][0]
                polygon = [(pt[0], pt[1]) for pt in coordinates]
                mask = fill_between(polygon)
                masks.append(mask)
                class_ids.append(1)  # Assuming class ID for 'building' is 1

        if masks:
            masks = np.stack(masks, axis=-1)
            class_ids = np.array(class_ids, dtype=np.int32)
        else:
            # If no masks, create a mask of zeros
            masks = np.zeros((image_info['height'], image_info['width'], 0), dtype=np.bool)
            class_ids = np.array([])

        return masks, class_ids
def train_model(dataset_dir, model_dir, coco_model_path, config, init_with="last", epochs=200):
    """
    Function to train the Mask RCNN model.

    Parameters:
    dataset_dir (str): Directory of the dataset.
    model_dir (str): Directory to save logs and trained model.
    coco_model_path (str): Path to COCO model weights.
    config (Config): Configuration for training.
    init_with (str): Which weights to start with: 'imagenet', 'coco', or 'last'.
    epochs (int): Number of training epochs.
    """
    # Load the training and validation datasets
    dataset_train = SpaceNetDataset()
    dataset_train.load_dataset(dataset_dir, 0, 3080)
    dataset_train.prepare()

    dataset_val = SpaceNetDataset()
    dataset_val.load_dataset(dataset_dir, 3081, 3850)
    dataset_val.prepare()

    # Create the model in training mode
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=model_dir)

    # Load weights
    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        model.load_weights(coco_model_path, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        model.load_weights(model.find_last(), by_name=True)

    # Train the model
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs,
                layers='all')


train_model(DATASET_DIR, MODEL_DIR, COCO_MODEL_PATH, config, init_with="coco", epochs=200)
