import os
import json
import logging
from pprint import pp

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

logger = logging.getLogger(__name__)

__all__ = ["register_meta_bdd100k"]

### classes ###
# 1: pedestrian
# 2: rider
# 3: car
# 4: truck
# 5: bus
# 6: train
# 7: motorcycle
# 8: bicycle
# 9: traffic light
# 10: traffic sign

BDD100K_ALL_CATEGORIES = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign"
]

METADATA = {"thing_classes": BDD100K_ALL_CATEGORIES}

def labels_loader(labels_path):
    with open(labels_path, "r") as f:
        labels = json.load(f)

    return labels

def load_bdd100k(name, metadata, json_file, image_root):
    # labels_path = "/scratch/wcheng/few-shot-object-detection/datasets/bdd100k/100k/labels/det_val_subset.json"
    gts = labels_loader(json_file)
    # pp(gts)
    # logger.debug(name)

    # dataset_path = "/scratch/wcheng/few-shot-object-detection/datasets/bdd100k/100k/frames"
    dataset = os.listdir(image_root)
    # logger.debug(dataset)

    data = []

    for file in dataset:
        labels = [frame for frame in gts["frames"] if frame["name"] == file][0]
        # logger.debug(metadata)
        annotation = {
            "file_name": os.path.join(image_root, file), # full path to image
            "image_id":  file, # image unique ID
            "height": 720, # height of image
            "width": 1280, # width of image
            "annotations": [
                {
                    "category_id": metadata["thing_classes"].index(label["category"]), # class unique ID
                    "bbox": [
                        label["box2d"]["x1"],
                        label["box2d"]["y1"],
                        label["box2d"]["x2"],
                        label["box2d"]["y2"]
                    ], # bbox coordinates
                    "bbox_mode": BoxMode.XYXY_ABS, # bbox mode, depending on your format
                } for label in labels["labels"]
            ]
        }

        data.append(annotation)

    return data

def register_meta_bdd100k(name, metadata, json_file, image_root, cfg, split):
    # register dataset (step 1)
    DatasetCatalog.register(
        name, # name of dataset, this will be used in the config file
        lambda: load_bdd100k( # this calls your dataset loader to get the data
            name, metadata, json_file, image_root # inputs to your dataset loader
        ),
    )

    # register meta information (step 2)
    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        cfg=cfg,
        evaluator_type="scalabel",
        **metadata,
        split=split
    )


# for debugging
if __name__ == "__main__":
    data = register_meta_bdd100k(name="bdd100k", obj_classes=BDD100K_ALL_CATEGORIES, metadata=METADATA)
    pp(data, sort_dicts=False)
