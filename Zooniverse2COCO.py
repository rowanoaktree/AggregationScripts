####
# Creator: Rowan Converse (rowanconverse@unm.edu)
# Date: 2022/08/02
# Purpose: Translate raw labels generated by volunteers on the participatory science platform Zooniverse into COCO format for public release 
# Ref COCO Camera Trap Standard: https://cocodataset.org/#format-data
####

#Load necessary modules
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from collections import OrderedDict 

#Load data
path = r"/Users/rowanconverse/Library/CloudStorage/OneDrive-UniversityofNewMexico/CV4Ecology/Prototyping/Data/Labels/drones-for-ducks-classifications.csv"
zooniverse = pd.read_csv(path)

###Annotations: import ID, Image ID, Category ID, bounding boxes (x,y, width, height). 
images = {}
annos = []
categories = {}

#This adds in labeler info-- omit for public release
#labelers = {}
for i in range(len(zooniverse)):
#This adds in labeler info-- omit for public release
#  labeler = zooniverse.user_name[i]
#  if labeler not in labelers:
#    labelers[labeler] = len(labelers) + 1
#  labeler_id = labelers[labeler]

  image_id = None
  try:
    imgrow = json.loads(zooniverse.subject_data[i])
  except:
    print('DEBUG')
    continue
  for key in imgrow.keys():
      try: 
        name = imgrow[key]["Filename"]
      except:
        print("DEBUG")
        continue
      if name not in images:
        images[name] = len(images) + 1
      
      image_id = images[name]
  try:
    row = json.loads(zooniverse["annotations"][i])
  except:
    print("DEBUG")
    continue
  for j in range(len(row)):
    if row[j]['task'] != 'T1':
        # task was not to draw a bounding box
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
      area = w*h

      if label not in categories:
        # label class has not yet been registered; add
        categories[label] = len(categories) + 1
      category_id = categories[label]
      annotation = {
        'annotation_id': len(annos)+1,
        'bbox': bbox,
        'area': area,
        'category_id': category_id,
        'image_id': image_id,
        'labeler_id': labeler_id
      }
      annos.append(annotation)

with open("zooniverse_coco.json", "w") as outfile:
    json.dump(annos, outfile)