files_in = [
    r"/datadrive/Data/train_filter2.csv",
    r"/datadrive/Data/val_filter2.csv",
    r"/datadrive/Data/test_filter2.csv"
]

files_out = [
    r"/datadrive/Data/train_filter3.csv",
    r"/datadrive/Data/val_filter3.csv",
    r"/datadrive/Data/test_filter3.csv"
]

img_size = [684, 521]

for f in range(len(files_in)):
    with open(files_in[f], 'r') as fin, open(files_out[f], 'w') as fout:
        fout.write(fin.readline())
        while True:
            line = fin.readline()
            if not len(line):
                break
            tokens = line.strip().split(',')
            bbox = [float(f) for f in tokens[1:5]]
            bbox[2] = min(bbox[2], img_size[0])
            bbox[3] = min(bbox[3], img_size[1])
            fout.write(','.join([
                tokens[0],
                *[str(b) for b in bbox],
                *tokens[5:]
            ]) + '\n')