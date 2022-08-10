from pybboxes import BoundingBox
import pandas as pd

f = r"/Users/rowanconverse/Library/CloudStorage/OneDrive-UniversityofNewMexico/CV4Ecology/Prototyping/Data/Labels/train/train.json"
with open(f, encoding='utf-8') as inputfile:
    df = pd.read_json(inputfile)

bboxes, filenames, cat_ids = df["bbox"].to_list(), df["filename"].to_list(), df["category_id"].to_list()
vocboxes = []
for ann in range(len(bboxes)):
    img = filenames[ann]
    bbox = bboxes[ann]
    if bbox[2] <= 1 or bbox[3] <= 1:
        continue
    try:
        coco_bbox = BoundingBox.from_coco(*bbox, strict=False)  # <[98 345 322 117] (322x117) | Image: (?x?)>
        voc_bbox = list(coco_bbox.to_voc(return_values=True))
    except:
        print("DEBUG")
        continue
    
    label = cat_ids[ann]
    vocanno = {
    "filename": img,
    "bbox": voc_bbox,
    "label": label
    }
    vocboxes.append(vocanno)

#Save VOC boxes to csv
labelpath = r"/Users/rowanconverse/Library/CloudStorage/OneDrive-UniversityofNewMexico/CV4Ecology/Prototyping/Data/Labels/train/train.csv"

with open(labelpath, 'w') as f:
    f.write('image_path, xmin, ymin, xmax, ymax, label')
    for item in vocboxes:
        f.write('{}, {}, {}\n'.format(
            item["filename"],
            ', '.join([str(b) for b in item["bbox"]]),
            item["label"]
        ))
print("done")