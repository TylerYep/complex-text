import matplotlib
import numpy as np
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

data = dict()
loss = []
epoch = 0
next_5 = 0
current_run = 'numrs 1bate 32embm 16hidm 64lea.001'
for file_name in ['results1.txt', 'results2.txt']:
    with open('analysis/' + file_name) as f:
        for line in f:
            if next_5 > 0:
                current_run += line[:3] + line[-4:-1]
                next_5 -= 1

            if line.find("num layers ") >= 0:
                x = [a[0] for a in loss]
                y = [b[1] for b in loss]

                if len(x) > 12:
                    while len(x) <= 50:
                        x.append(epoch)
                        y.append(y[-1])
                        epoch += 1
                    data[current_run] = (x,y)


                next_5 = 4
                current_run = line[:3] + line[-4:-1]
                loss = []
                epoch = 0

            dev_acc_ind = line.find("Best Dev Accuracy: ")
            if dev_acc_ind >= 0:
                dev_acc = float(line[dev_acc_ind + len("Best Dev Accuracy: "): -1])
                loss.append((epoch, dev_acc))
                epoch += 1


for current_run in data:
    x, y = data[current_run]
    print(current_run)
    plt.plot(x, y, label=current_run, c=np.random.rand(3,))

print(data.keys())
axes = plt.gca()
axes.set_xlim([0,50])
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Percentage accuracy')
plt.title('LSTM Classification Accuracy w/ Different Hyperparameters')
plt.show()

# loss = []
# epoch = 0
# next_5 = 0
# current_run = ''
# for file_name in ['results1.txt', 'results2.txt']:
#     with open('analysis/' + file_name) as f:
#         for line in f:
#             if next_5 > 0:
#                 current_run += line[:-1] + ', '
#                 next_5 -= 1
#
#             if line.find("num layers ") >= 0:
#                 print(current_run)
#                 next_5 = 4
#                 current_run = line[:-1] + ', '
#                 x = [a[0] for a in loss]
#                 y = [b[1] for b in loss]
#                 if len(x) > 40:
#                     plt.plot(x, y, label=current_run)
#                 loss = []
#                 epoch = 0
#             train_acc_ind = line.find("Train Accuracy: ")
#             if train_acc_ind >= 0:
#                 dev_acc_ind = line.find("Dev Accuracy: ")
#                 train_acc = float(line[train_acc_ind + len("Train Accuracy: "):-1])
#                 dev_acc = float(line[dev_acc_ind + len("Dev Accuracy: "): train_acc_ind-2])
#
#                 loss.append((epoch, train_acc, dev_acc))
#                 epoch += 1
#
# axes = plt.gca()
# axes.set_xlim([0,50])
# plt.legend()
# plt.show()
