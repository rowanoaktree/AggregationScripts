import pandas as pd
import torch
from torchvision.ops import nms
from tqdm import trange


def quick_dirty_nms(df, iou_threshold=0.05):
    labelclasses = df['label'].unique()
    imgs = df['image_path'].unique()

    pbar = trange(len(labelclasses)*len(imgs))

    df_out = []

    for lc in labelclasses:
        lc_subset = df[df.label == lc]

        for img in imgs:
            subset = lc_subset[lc_subset.image_path == img]
            boxes = torch.tensor([
                subset.xmin.to_list(),
                subset.ymin.to_list(),
                subset.xmax.to_list(),
                subset.ymax.to_list()
            ]).T
            scores = torch.ones(size=(len(boxes),), dtype=torch.float32)
            idx = nms(boxes, scores, iou_threshold).numpy()
            df_out.append(subset.iloc[idx])

            pbar.set_description(f'Labelclass: {lc}; img: {img}')
            pbar.update(1)

    pbar.close()

    return pd.concat(df_out)


if __name__ == '__main__':
    iou_threshold = 0.5     # smaller values lead to more aggressive NMS
    csv_file = r"/datadrive/Data/val_filter4.csv" 
    df = pd.read_csv(csv_file)

    df_nms = quick_dirty_nms(df, iou_threshold)

    df_nms.to_csv("/datadrive/Data/val_filter4_nms_{}.csv".format(
        str(iou_threshold).replace('.', '')
    ))