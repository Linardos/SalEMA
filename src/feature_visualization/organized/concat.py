import cv2
import numpy as np
import os
import sys
print(sys.version)
layer_num = 17
salema = "./SalEMA-L{}".format(layer_num)
salclstm = "./SalCLSTM-L{}".format(layer_num)

emafiles = os.listdir(salema)
clstmfiles = os.listdir(salclstm)

elist = []
clist = []

for e in emafiles:
    temp = cv2.imread(os.path.join(salema, e), cv2.COLOR_BGR2GRAY)
    elist.append(temp)

e_im = np.concatenate(tuple(elist), 1)


for c in clstmfiles:
    temp = cv2.imread(os.path.join(salclstm, c), cv2.COLOR_BGR2GRAY)
    clist.append(temp)

c_im = np.concatenate(tuple(clist), 1)

vertical = np.concatenate((e_im, c_im), 0)
cv2.imwrite("concatenated{}.png".format(layer_num)
, vertical)
