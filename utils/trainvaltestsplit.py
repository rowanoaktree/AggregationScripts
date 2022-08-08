import pandas as pd
import json
from sklearn.model_selection import train_test_split
path = r"/Users/rowanconverse/Library/CloudStorage/OneDrive-UniversityofNewMexico/CV4Ecology/Prototyping/Data/Labels/snippet_coco.json"
cocozoo = json.loads(path)

#Just a simple random split, 60% train, 20% validation, 20% test

train, test = train_test_split(cocozoo, test_size=0.4)
val, test = train_test_split(test, test_size=0.5)

with open("train.json", "w") as outfile:
    json.dump(train, outfile)

with open("val.json", "w") as outfile:
    json.dump(val, outfile)

with open("test.json", "w") as outfile:
    json.dump(test, outfile)