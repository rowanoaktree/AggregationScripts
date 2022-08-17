'''
    2022 Benjamin Kellenberger
'''

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from deepforest import preprocess


class BirdsDataset(Dataset):

    def __init__(self, data_root, csv_file, label_dict):
        super(BirdsDataset, self).__init__()

        self.data_root = data_root

        # load data
        self.data = []
        meta = pd.read_csv(csv_file)
        img_paths = meta['image_path'].unique()
        for ip in img_paths:
            rows = meta[meta['image_path'] == ip]
            bboxes = np.array([
                rows['xmin'].to_list(),
                rows['xmax'].to_list(),
                rows['ymin'].to_list(),
                rows['ymax'].to_list()
            ])
            labels = rows['label'].to_list()
            self.data.append((
                ip,
                {
                    'boxes': torch.from_numpy(bboxes),
                    'labels': torch.tensor([label_dict[l] for l in labels]).long()
                }
            ))



    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        imgPath, labels = self.data[idx]

        img = np.array(Image.open(os.path.join(self.data_root, imgPath)).convert("RGB")).astype("float32")
        img = preprocess.preprocess_image(img, device='cpu')

        return img, labels
