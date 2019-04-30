import cv2
import numpy as np
import os
import sys
print(sys.version)

VID = "actioncliptest00474_5"
def main():
    horizontal = []

    hollyfolder = os.listdir("/home/linardos/Hollywood-2/testing/{}/images".format(VID))
    hollypredfolder = os.listdir("/home/linardos/Hollywood-2/testing/{}/SalEMA30D_H_predictions".format(VID))
    number_of_frames = len(hollyfolder)
    print("Number of frames is {}".format(number_of_frames))
    for i in range(1, number_of_frames+1):
        """
        frame = cv2.imread("/home/linardos/Documents/QualAnal/frames/{}/{}.png".format(VID, i), cv2.COLOR_BGR2GRAY)
        salema = cv2.imread("/home/linardos/Documents/QualAnal/SalEMA30/0{}/{}.png".format(VID, str(i).zfill(4)))
        salclstm = cv2.imread("/home/linardos/Documents/QualAnal/SalCLSTM30/0{}/{}.png".format(VID, str(i).zfill(4)))
        salbce = cv2.imread("/home/linardos/Documents/QualAnal/SBCE/0{}/{}.png".format(VID, str(i).zfill(4)))
        gt = cv2.imread("/home/linardos/Documents/QualAnal/maps/{}/{}.png".format(VID, str(i).zfill(4)))
        gt = cv2.resize(gt, (salema.shape[1], salema.shape[0]), interpolation=cv2.INTER_AREA)
        """
        frame = cv2.imread("/home/linardos/Hollywood-2/testing/{}/images/{}".format(VID, hollyfolder[i-1]), cv2.COLOR_BGR2GRAY)
        salema = cv2.imread("/home/linardos/Hollywood-2/testing/{}/SalEMA30D_H_predictions/{}".format(VID, hollypredfolder[i-1]), cv2.COLOR_BGR2GRAY)
        salclstm = cv2.imread("/home/linardos/Hollywood-2/testing/{}/SalGANmid_H_predictions/{}".format(VID, hollypredfolder[i-1]), cv2.COLOR_BGR2GRAY)
        gt = cv2.imread("/home/linardos/Hollywood-2/testing/{}/maps/{}".format(VID, hollyfolder[i-1]), cv2.COLOR_BGR2GRAY)
        gt = cv2.resize(gt, (salema.shape[1], salema.shape[0]), interpolation=cv2.INTER_AREA)

        one = ProduceOverlayed(frame, salema, "salema", i)
        two = ProduceOverlayed(frame, salclstm, "salclstm", i)
        #thr = ProduceOverlayed(frame, salbce, "salbce", i)
        fou = ProduceOverlayed(frame, gt, "gts", i)

        # uncommenct this for concatenation
        temp = [one, two, fou]
        new_im = np.concatenate((one,two,fou), 1)
        horizontal.append(new_im)
        #cv2.imshow("fin", new_im)
        #cv2.waitKey(0)
        if i%50==0:
            print("Frame {} done.".format(i))

    try:
        vertical = np.concatenate(tuple(i for i in horizontal), 0)
        cv2.imwrite("./{}qa.png".format(VID), vertical)
    except:
        print("Not enough frames for vertical concatenation")
        cv2.imwrite("./{}qa.png".format(VID), horizontal[0])



def ProduceOverlayed(X, prediction, model_name, i):

    if not os.path.exists("./{}/{}".format(model_name, VID)):
        os.makedirs("./{}/{}".format(model_name, VID))

    Y = cv2.applyColorMap(prediction, cv2.COLORMAP_JET)
    X = cv2.resize(X, (Y.shape[1], Y.shape[0]), interpolation=cv2.INTER_AREA)
    fin = cv2.addWeighted(Y, 0.5, X, 0.5, 0)

    cv2.imwrite("./{}/{}/{}.png".format(model_name, VID, i), fin)

    return(fin)

main()
