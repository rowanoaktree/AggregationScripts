####
# Creator: Rowan Converse (rowanconverse@unm.edu)
# Date: 2022/08/02
# Purpose: Translate raw labels generated by USFWS biologists in Labebox into COCO format for public release 
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
path = r"/Users/rowanconverse/Library/CloudStorage/OneDrive-UniversityofNewMexico/CV4Ecology/Prototyping/Data/Labels/zooniverse_snippet.csv"
zooniverse = pd.read_csv(path)

###Info 
whattimeisitrightnowdotcom = datetime.date.today()
year = {"year": 2022}
vers = {"version": "1.0"}
desc = {"description": "This dataset includes annotations of UAS imagery collected x to y , 2018, at Bosque del Apache National Wildlife Refuge in New Mexico. Over 3,000 volunteers identified waterfowl in 611 images using three morphological categories (Duck, Goose, Crane). The labels were collected via the participatory science platform, Zooniverse. Contact Rowan Converse (rowanconverse@unm.edu) with questions about this dataset. Please cite using a CC-By license with ASPIRE as the data repository."}
contr = {"contributor": "Center for the Advancement of Spatial Informatics Research and Education (ASPIRE), University of New Mexico; Project Manager Rowan Converse"}
url = {"url": "https://aspire.unm.edu/projects/project/ducks-and-drones.html"}
date = {"date created": whattimeisitrightnowdotcom}

infolist = [year, vers, desc, contr, url, date]

info = {"info": infolist}

###Images

#Derive list of images
for i in range(len(usfws)):
  imglist = list(set(img['External ID'] for img in usfws))

#Add unique IDs to each filename
imgIDs = [{v: k for k, v in enumerate(
   OrderedDict.fromkeys(imglist), 1)}
      [n] for n in imglist]
img = dict(zip(imgIDs, imglist))
  
###Annotations: import ID, Image ID, Category ID, bounding boxes (x,y, width, height). 
 
#categories: use code for species list (name), match with integer values (id) (NOT ZERO)
#Categories
spplist = []
for i in range(len(zooniverse)):
  row = json.loads(zooniverse.annotations[i])
  for j in range(len(row)):
    annlist = row[j]['value']
    for k in range(len(annlist)):
      ann = annlist[k]
      cur_species = ann["tool_label"] 
      spplist.append(cur_species)
spp = set(spplist)

sppIDs = [{v: k for k, v in enumerate(
   OrderedDict.fromkeys(spplist), 1)}
      [n] for n in spp]
cat = dict(zip(sppIDs, spp))
categories = {"categories": cat}
categories

#also add labeler info
userlist = list(set(zooniverse.user_id))
userIDs = [{v: k for k, v in enumerate(
   OrderedDict.fromkeys(userlist), 1)}
      [n] for n in userlist]
labelers = dict(zip(userIDs, userlist))


###License
lic_id = {"id": 1} 
lic_name = {"name": "Creative Commons (CC)-BY"}
lic_url = {"url": "https://creativecommons.org/about/cclicenses/"}
licenselist = [lic_id, lic_name, lic_url]
license = {"license": licenselist}

#Finally, merge all the dictionaries into one JSON file and save it to the data directory
zooniversecoco = json_dump(info, img, annos, license)