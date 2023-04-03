import json
import pandas as pd
import datetime
import ast 

# Load the data from the CSV file
path = "/Users/rowanconverse/Library/CloudStorage/OneDrive-UniversityofNewMexico/CV4Ecology/Prototyping/Data/Labels/coco/labelbox.csv"
df = pd.read_csv(path)
df["bbox"] = df["bbox"].apply(ast.literal_eval)

# Create dictionaries to store the mapping between labels and IDs
categories = {}
labelers = {}

# Initialize variables for creating the COCO JSON format
images = {}
annos = []

# Iterate over each row in the DataFrame
for i in range(len(df)):
    # Extract the image filename
    filename = df["filename"][i]

    # If the filename has not yet been seen, add it to the images dictionary
    if filename not in images:
        images[filename] = {
            "id": len(images) + 1,
            "file_name": filename
        }

    # Extract the bounding box information
    bbox = df["bbox"][i]
    xmin, ymin, width, height = bbox

    # Extract the labeler information
    labeler = df["labeler"][i]

    # If the labeler has not yet been seen, add it to the labelers dictionary
    if labeler not in labelers:
        labelers[labeler] = len(labelers) + 1

    # Extract the category information
    category = df["category"][i]

    # If the category has not yet been seen, add it to the categories dictionary
    if category not in categories:
        categories[category] = len(categories) + 1

    # Create the annotation dictionary
    annotation = {
        "id": df["annotation_ID"][i],
        "image_id": images[filename]["id"],
        "category_id": categories[category],
        "bbox": [xmin, ymin, width, height],
        "area": width * height,
        'iscrowd': 0,
        "labeler_id": labelers[labeler]
    }

    # Add the annotation to the list of annotations
    annos.append(annotation)

# Create the COCO JSON format
coco_output = {
    "images": list(images.values()),
    "annotations": annos,
    "categories": [{"id": categories[k], "name": k} for k in categories],
}

for anno in annos:
    anno['id'] = int(anno['id'])
    anno['image_id'] = int(anno['image_id'])
    anno['category_id'] = int(anno['category_id'])
    anno['bbox'] = [float(x) for x in anno['bbox']]

# Save the output file
with open(datetime.datetime.now().strftime('%Y%m%d') + '_dronesforducks_raw_experts.json', "w") as outfile:
    json.dump(coco_output, outfile)