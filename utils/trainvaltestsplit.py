import pandas as pd
import json
import numpy as np
import os
path = r"/Users/rowanconverse/Library/CloudStorage/OneDrive-UniversityofNewMexico/CV4Ecology/Prototyping/Data/Labels/snippet_coco.json"
annotations = pd.read_csv(path)

#Just a simple random split, 60% train, 20% validation, 20% test, making sure there is no image overlap between sets

image_paths = annotations["image_path"].unique()

valid_paths = np.random.choice(image_paths, int(len(image_paths)*0.4) )
valid_annotations = annotations.loc[annotations["image_path"].isin(valid_paths)]
train_annotations = annotations.loc[~annotations["image_path"].isin(valid_paths)]
test_paths = np.random.choice(valid_paths, int(len(valid_paths)*0.5))
test_annotations = valid_annotations.loc[valid_annotations["image_path"].isin(test_paths)]
valid_annotations = valid_annotations.loc[~valid_annotations["image_path"].isin(test_paths)]

#Checking that there is no overlap between sets
trainfiles = set(train_annotations["image_path"])
valfiles = set(valid_annotations["image_path"])
testfiles = set(test_annotations["image_path"])
print(set(testfiles).intersection(valfiles))
print(set(trainfiles).intersection(valfiles))
print(set(set(testfiles).intersection(trainfiles)))

#Make sure the total split numbers look good
print(len(annotations))
print(len(train_annotations))
print(len(valid_annotations))
print(len(test_annotations))

#Write to your directory (needs to be with the right image sets-- adjust this in the final version)
lbl_dir = "/Users/rowanconverse/Library/CloudStorage/OneDrive-UniversityofNewMexico/CV4Ecology/Prototyping/Data/Labels"
annotations_file= os.path.join(lbl_dir,"train.csv")
validation_file= os.path.join(lbl_dir,"valid.csv")
test_file = os.path.join(lbl_dir, "test.csv")

train_annotations.to_csv(annotations_file,index=False)
valid_annotations.to_csv(validation_file,index=False)
test_annotations.to_csv(test_file,index=False)