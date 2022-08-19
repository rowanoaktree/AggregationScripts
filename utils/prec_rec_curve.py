import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def precRecCurve(df_pred, df_evaluated, num_steps=50, agnostic_label=None):
    if agnostic_label is not None:
        df_pred['label'] = agnostic_label
        df_evaluated['true_label'] = agnostic_label
        df_evaluated.predicted_label[np.logical_not(pd.isnull(df_evaluated['predicted_label']))] = agnostic_label

    
    # get min-max range of confidence scores
    minConf = df_evaluated['score'].min()
    maxConf = df_evaluated['score'].max()
    steps = np.linspace(minConf, maxConf, num_steps)
    labelclasses = df_evaluated['true_label'].unique()

    result = {}
    for labelclass in labelclasses:
        prec, rec = [], []

        num_nan = np.sum((df_evaluated['true_label'] == labelclass) & (np.isnan(df_evaluated['score'])))

        for step in steps:
            rows = df_evaluated[df_evaluated['score'] >= step]
            tp = np.sum((rows['predicted_label'] == labelclass) & (rows['true_label'] == labelclass))
            fp = np.sum((rows['predicted_label'] == labelclass) & (rows['true_label'] != labelclass)) + \
                        np.sum((df_pred.score >= step) & (df_pred.label == labelclass))
            fn = np.sum((rows['predicted_label'] != labelclass) & (rows['true_label'] == labelclass)) + num_nan

            try:
                p = tp/(tp+fp)
            except:
                p = 0
            try:
                r = tp/(tp+fn)
            except:
                r = 0
            prec.append(p)
            rec.append(r)
        
        result[labelclass] = {
            'precision': prec,
            'recall': rec
        }
    return result


def save_precrec_plot(precRec, destination):
    plt.figure()
    for species in precRec:
        plt.plot(precRec[species]['recall'], precRec[species]['precision'], label=species)
        plt.draw()
    plt.title('Precision-recall curve')
    plt.xlabel('recall')
    plt.xlim([0, 1])
    plt.ylabel('precision')
    plt.ylim([0, 1])
    plt.grid()
    plt.legend()
    plt.savefig(destination)


# if __name__ == '__main__':
#     import os
#     import pandas as pd
#     import matplotlib.pyplot as plt

#     save_dir = r"predictions/val"
#     preds = pd.read_csv('temp.csv')
#     precRec = precRecCurve(preds)
#     save_precrec_plot(precRec, os.path.join(save_dir, 'prec_rec.png'))

#     # class-agnostic
#     preds['true_label'] = 'bird'
#     preds.predicted_label[np.logical_not(pd.isnull(preds['predicted_label']))] = 'bird'
#     precRec = precRecCurve(preds)
#     save_precrec_plot(precRec, os.path.join(save_dir, 'prec_rec_speciesAgnostic.png'))
