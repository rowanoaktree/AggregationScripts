import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from collections import OrderedDict

# Load data
path = r"/Users/rowanconverse/Library/CloudStorage/OneDrive-UniversityofNewMexico/CV4Ecology/Prototyping/Data/Labels/originals/20230227_dgc.csv"
zooniverse = pd.read_csv(path)

# Annotations: import ID, Image ID, Category ID, bounding boxes (x,y, width, height).
images = []
annotations = []
categories = []

for i in range(len(zooniverse)):
    image_id = None
    try:
        imgrow = json.loads(zooniverse.subject_data[i])
    except:
        print('DEBUG')
        continue
    for key in imgrow.keys():
        try:
            filename = imgrow[key]["Filename"]
        except:
            print("DEBUG")
            continue
        if filename not in images:
            images.append({
                'id': len(images) + 1,
                'file_name': filename
            })
        image_id = images[-1]['id']
    try:
        row = json.loads(zooniverse["annotations"][i])
    except:
        print("DEBUG")
        continue
    for j in range(len(row)):
        if row[j]['task'] != 'T1':
            # Task was not to draw a bounding box
            continue

        annlist = row[j]['value']
        for k in range(len(annlist)):
            if k == "null":
                continue
            ann = annlist[k]
            try:
                x = ann["x"]
            except:
                print("DEBUG")
                continue
            y = ann["y"]
            try:
                w = ann["width"]
            except:
                print("DEBUG")
                continue
            h = ann["height"]
            label = ann["tool_label"]
            bbox = [x, y, w, h]
            area = w * h

            category_id = None
            for cat in categories:
                if cat['name'] == label:
                    category_id = cat['id']
                    break
            if category_id is None:
                # Label class has not yet been registered; add
                category_id = len(categories) + 1
                categories.append({
                    'id': category_id,
                    'name': label
                })

            annotation = {
                'id': len(annotations) + 1,
                'image_id': image_id,
                'category_id': category_id,
                'bbox': bbox,
                'area': area,
                'iscrowd': 0
            }
            annotations.append(annotation)

with open(datetime.datetime.now().strftime('%Y%m%d') + '_dronesforducks.json', "w") as outfile:
    json.dump({"images": images,
               "annotations": annotations,
               "categories": categories,
              }, outfile)