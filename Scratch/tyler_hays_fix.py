#Load necessary modules
import json
from UliEngineering.Math.Coordinates import BoundingBox
import numpy as np
import pandas as pd

#Load JSON file of Labelbox labels
path = r"/Users/rowanconverse/Library/CloudStorage/OneDrive-UniversityofNewMexico/CV4Ecology/Prototyping/Data/Labels/originals/labelbox.json"
with open(path) as f:
  usfws = json.load(f)

annos = []

for i in range(len(usfws)):
    #Check if the image was skipped. If Skipped==True there is no values in 'Label' key.
    if usfws[i]["Skipped"]==False:
        label = usfws[i]["Label"]
        for x in label:
            name = x
        for value in label.values():
            for k in value:
                for j in k.values(): 
                        #Create temporary list
                        polycoords=[]
                        for l in j:
                            pairs = list(l.values())
                            #Append all of the pairs to the temp list
                            polycoords.append(pairs)
                        #When all pairs are in temp list, turn the list into a numpy array.
                        polycoords=np.array(polycoords)
                        #BoundingBox function expects numpy array.
                        minbox=BoundingBox(polycoords)
                        bbox = [minbox.minx, minbox.miny, minbox.width, minbox.height]
                
                        annotation = {
                        "annotation_ID": len(annos)+1,
                        "bbox": list(bbox),
                        "filename": usfws[i]["External ID"],
                        "labeler": usfws[i]["Created By"],
                        "category": x
                        }
                        annos.append(annotation)  

df = pd.DataFrame(annos)
df.to_csv('labelbox.csv')
