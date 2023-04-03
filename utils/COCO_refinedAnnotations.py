import json
import pandas as pd
import ast

# Load CSV data
path = "/Users/rowanconverse/Library/CloudStorage/OneDrive-UniversityofNewMexico/Dissertation/1_Chapter/consensus/data/crowdsourced/20230307_zooniverseconsensuslabels.csv"
data = pd.read_csv(path)
data["bbox"] = data["bbox"].apply(ast.literal_eval)

# Initialize variables for COCO JSON format
images = []
annotations = []
categories = []

# Loop through data and create COCO format annotations and categories
for index, row in data.iterrows():
    # Extract relevant data from CSV
    annotation_id = row["annotation_id"]
    image_name = row["filename"]
    category_name = row["category"]
    bbox = row["bbox"]
    category_id = row["category_id"]
    
    # Check if category already exists, if not add it
    if category_id not in [cat["id"] for cat in categories]:
        categories.append({"id": category_id, "name": category_name})
    
    # Create annotation dictionary and append to annotations list
    annotation = {
        "id": annotation_id,
        "image_id": None,
        "category_id": category_id,
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "iscrowd": 0
    }
    annotations.append(annotation)
    
    # Check if image already exists, if not add it
    if image_name not in [img["file_name"] for img in images]:
        image_id = len(images) + 1
        images.append({"id": image_id, "file_name": image_name})
        annotation["image_id"] = image_id
    else:
        image_id = [img["id"] for img in images if img["file_name"] == image_name][0]
        annotation["image_id"] = image_id

# Create dictionary for final COCO JSON output
coco_output = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

# Save output as JSON file with desired name format
today = pd.Timestamp.today().strftime("%Y%m%d")
filename = today + "_dronesforducks_refined.json"
with open(filename, "w") as outfile:
    json.dump(coco_output, outfile)