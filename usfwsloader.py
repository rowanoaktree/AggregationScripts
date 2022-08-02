import json

path = r"/Users/rowanconverse/Library/CloudStorage/OneDrive-UniversityofNewMexico/CV4Ecology/Prototyping/Data/Labels/labelbox.json"
with open(path) as f:
  usfws = json.load(f)

#If you were trying to get all images in a folder:
# import os
# import glob
# glob.glob(input_folder, recursive = True)