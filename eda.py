from matplotlib import pyplot as plt
import os
import numpy as np
from PIL import Image
import pandas as pd

fig = plt.figure(figsize=(20, 8))
# display 20 images
labels = pd.read_csv("train_labels.csv")
train_imgs = os.listdir('../autodl-tmp/data/train')
test_imgs = os.listdir('../autodl-tmp/data/test')
print(len(train_imgs))
print(len(test_imgs))

for idx, img in enumerate(np.random.choice(train_imgs, 8)):
    # print(img)
    ax = fig.add_subplot(2, 8//2, idx+1, xticks=[], yticks=[])
    im = Image.open('../autodl-tmp/data/train/' + img)
    plt.imshow(im)
    lab = labels.loc[labels['id'] == img.split('.')[0], 'label'].values[0]
    ax.set_title('Label: %s'%lab)

plt.savefig("vis.png")