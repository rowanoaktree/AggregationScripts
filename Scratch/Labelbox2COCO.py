import labelbox
from labelbox.data.annotation_types import Label, LabelList, ImageData, Point, ObjectAnnotation, Rectangle, Polygon
from labelbox.data.serialization import COCOConverter

project = Client.get_project('PROJECT_ID')
labels = project.label_generator()

mask_path = "./masks/"
image_path = './images/'

coco_labels = COCOConverter.serialize_panoptic(
    labels,
    image_root=image_path,
    mask_root=mask_path,
    ignore_existing_data=True
)

