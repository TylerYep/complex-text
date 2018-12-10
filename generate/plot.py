import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

def arr_from_txt(fname):
    with open(fname, 'r') as f:
        text = f.readlines()
        text = [t.split() for t in text]
        text = [t for t in text if t[0] == 'Loss']
        arr = [float(t[-1]) for t in text]
        return arr

c = ['r', 'b', 'k']
def line(arr, label):
    #plt.scatter(range(len(arr)), arr, s=1)
    plt.plot(arr, label=label, color=c[int(label[-1])-2])

labels = ['level {}'.format(i) for i in [2, 3, 4]]
arrs = [arr_from_txt('lvl{}.txt'.format(x)) for x in [2, 3, 4]]
min_len = min([len(x) for x in arrs])
arrs = [x[:min_len] for x in arrs]

for a, l in zip(arrs, labels): line(a, l)
plt.title('Training the generation models')
plt.xlabel('Epoch')
plt.ylabel('Average Cross Entropy Loss')
plt.legend()
plt.show()



