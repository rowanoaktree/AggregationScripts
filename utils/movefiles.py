import os
from os import path
import shutil
import pandas as pd
import csv

src = "/datadrive/Data/train"
dst = "/datadrive/Data/subset"
train = pd.read_csv("/Users/rowanconverse/Library/CloudStorage/OneDrive-UniversityofNewMexico/CV4Ecology/Prototyping/Data/Labels/train.csv")

img = train["image_path"]
images = set(train['image_path'].tolist())
s = sorted(images)

for filename in os.listdir(src):
    if filename in s:
        filename = os.path.join(src, filename)
        shutil.copy(filename, dst)
        print(str(filename)+" has been copied")
    else:
        print(str(filename)+" file is not in this directory")