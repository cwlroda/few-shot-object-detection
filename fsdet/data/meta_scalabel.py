### format ###
# def dataset_loader(name, thing_classes):
#     data = []
#     for file in dataset:
#         annotation = [
#             "file_name" : "x.png", # full path to image
#             "image_id" :  0, # image unique ID
#             "height" : 123, # height of image
#             "width" : 123, # width of image
#             "annotations": [
#                 "category_id" : thing_classes.index("class_name"), # class unique ID
#                 "bbox" : [0, 0, 123, 123], # bbox coordinates
#                 "bbox_mode" : BoxMode.XYXY_ABS, # bbox mode, depending on your format
#             ]
#         ]
#         data.append(annotation)
#     return data

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

import os
import json
from pprint import pp

from detectron2.structures import BoxMode

SCALABEL_ALL_CATEGORIES = [
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

def labels_loader(labels_path):
    with open(labels_path, "r") as f:
        labels = json.load(f)

    return labels

def dataset_loader(name=None, obj_classes=None):
    labels_path = "/scratch/wcheng/few-shot-object-detection/datasets/scalabel/labels.json"
    labels = labels_loader(labels_path)

    dataset_path = "/scratch/wcheng/few-shot-object-detection/datasets/scalabel/img"
    dataset = os.listdir(dataset_path)

    url = "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/"

    data = []
    count = 0

    for file in dataset:
        metadata = [frame for frame in labels["frames"] if frame["name"] == (url + file)]
        if metadata:
            metadata = metadata[0]
        else:
            continue

        annotation = {
            "file_name": os.path.join(dataset_path, file), # full path to image
            "image_id":  count, # image unique ID
            "height": 1280, # height of image
            "width": 720, # width of image
            "annotations": [
                {
                    "category_id": SCALABEL_ALL_CATEGORIES.index(anno["category"]), # class unique ID
                    "bbox": [
                        anno["box2d"]["x1"],
                        anno["box2d"]["y1"],
                        anno["box2d"]["x2"],
                        anno["box2d"]["y2"]
                    ], # bbox coordinates
                    "bbox_mode": BoxMode.XYXY_ABS, # bbox mode, depending on your format
                } for anno in metadata["labels"]
            ]
        }

        data.append(annotation)
        count += 1

    return data

if __name__ == "__main__":
    data = dataset_loader()
    pp(data, sort_dicts=False)
