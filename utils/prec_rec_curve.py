import numpy as np

def precRecCurve(df, num_steps=50):
    # get min-max range of confidence scores
    minConf = df['score'].min()
    maxConf = df['score'].max()
    steps = np.linspace(minConf, maxConf, num_steps)
    labelclasses = df['true_label'].unique()

    result = {}
    for labelclass in labelclasses:
        prec, rec = [], []

        for step in steps:
            rows = df[df['score'] >= step]
            tp = np.sum((rows['predicted_label'] == labelclass) & (rows['true_label'] == labelclass))
            fp = np.sum((rows['predicted_label'] == labelclass) & (rows['true_label'] != labelclass))
            fn = np.sum((rows['predicted_label'] != labelclass) & (rows['true_label'] == labelclass))

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