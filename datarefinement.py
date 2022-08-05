import pandas as pd
path = r"/Users/rowanconverse/Library/CloudStorage/OneDrive-UniversityofNewMexico/CV4Ecology/Prototyping/Data/Labels/zooniverse_coco.json"
cocozoo = pd.read_csv(path)

#First step: iterate over each bounding box
#calculate IOU between each box and all the other boxes (in the same image)
#for those that intersect-- preserve the minimum area
#remove any boxes that have no intersection with other boxes (or flag for manual review?)
#finally, for the new minimum area box: assign a label that is the plurality vote of the label id for each box that made up the intersection area for it