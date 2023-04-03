import json
import pandas as pd
from collections import OrderedDict
import datetime
import ast

# load data
path = "/Users/rowanconverse/Library/CloudStorage/OneDrive-UniversityofNewMexico/Dissertation/1_Chapter/consensus/data/expert/20230307_expertconsensuslabels_spponly.csv"
data = pd.read_csv(path)
data["bbox"] = data["bbox"].apply(ast.literal_eval)

# create a mapping of category names to category IDs
categories = {}
for cat in data["category"].unique():
    categories[cat] = len(categories) + 1

# process annotations
images = {}
annotations = []
for i in range(len(data)):
    img_file = data.iloc[i]["filename"]
    if img_file not in images:
        images[img_file] = {
            "id": len(images) + 1,
            "file_name": img_file
        }
    img_id = images[img_file]["id"]
    bbox = data.iloc[i]["bbox"]
    annotation = {
        "id": int(data.iloc[i]["annotation_ID"]),
        "image_id": img_id,
        "category_id": categories.get(data.iloc[i]["category"], 1),
        "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
        "iscrowd": 0
    }
    annotations.append(annotation)

# convert images dictionary to list
images = list(images.values())

# create COCO-style dictionary
coco_dict = OrderedDict()
coco_dict["info"] = {"year": 2023}
coco_dict["images"] = images
coco_dict["categories"] = [{"id": cat_id, "name": cat_name} for cat_name, cat_id in categories.items()]
coco_dict["annotations"] = annotations

# save as JSON file
filename = datetime.datetime.now().strftime("%Y%m%d") + "_dronesforducks_expert_refined.json"
with open(filename, "w") as outfile:
    json.dump(coco_dict, outfile)