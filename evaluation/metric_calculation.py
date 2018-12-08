from salience_metrics import auc_judd, auc_shuff, cc, nss, similarity, normalize_map
"""
DHF1K paper: "we  employ  five  classic  met-rics,  namely  Normalized  Scanpath  Saliency  (NSS),  Sim-ilarity Metric (SIM), Linear Correlation Coefficient (CC),AUC-Judd (AUC-J), and shuffled AUC (s-AUC).""
"""
import cv2
import os
import numpy as np
import pickle
import datetime

gt_directory = "/imatge/lpanagiotis/work/DHF1K/maps"
sm_directory = "/imatge/lpanagiotis/work/DHF1K/SGplus_predictions"

continue_calculations = False

if continue_calculations:
    with open('metrics.txt', 'rb') as handle:
        final_metric_list = pickle.load(handle)
else:
    final_metric_list = []


# The directories are named 1-1000 so it should be easy to iterate over them
def inner_worker(i, packed, gt_path, sm_path): #packed should be a list of tuples (annotation, prediction)

    gt, sm = packed
    ground_truth = cv2.imread(os.path.join(gt_path, gt),cv2.IMREAD_GRAYSCALE)
    saliency_map = cv2.imread(os.path.join(sm_path, sm),cv2.IMREAD_GRAYSCALE)

    #print(np.max(ground_truth))
    ground_truth = cv2.resize(ground_truth, (saliency_map.shape[1], saliency_map.shape[0]), interpolation=cv2.INTER_AREA)
    # some error seems to occur, particularly on the first 3 metrics after the resize. CC and SIM are calculated normally. Noticeably the maximum value of 255 changes for the ground truth. It makes sense that this confuses location based metrics that aim to compare fixation points (hence maximum value locations).

    ground_truth[ground_truth==np.max(ground_truth)]=255
    # Rescaling turned out to cause a mess to the distribution, which results in a mess for the distribution based metrics (CC , SIM). This is evident from the fact that before rescaling the mean is close to 9 but afterwards it's close to 0. It is apparent that rescaling saliency maps down and then up again causes issues and should be avoided.
    #ground_truth = (ground_truth-np.min(ground_truth))/(np.max(ground_truth)-np.min(ground_truth))
    #ground_truth = ground_truth*255

    saliency_map_norm = normalize_map(saliency_map) # The functions are a bit haphazard. Some have normalization within and some do not.

    # Rescaling seems to fix AUC behavior, but CC and SIM break and give very low values. Weird behavior..
    # Calculate metrics
    AUC_JUDD = auc_judd(saliency_map_norm, ground_truth)
    AUC_SHUF = auc_shuff(saliency_map_norm, ground_truth, ground_truth)
    # the other ones have normalization within:
    NSS = nss(saliency_map, ground_truth)
    CC = cc(saliency_map, ground_truth)
    SIM = similarity(saliency_map, ground_truth)

    return ( AUC_JUDD,
                    AUC_SHUF,
                    NSS,
                    CC,
                    SIM )

start = datetime.datetime.now().replace(microsecond=0)
for i in range(1,701):

    if i == 57: #Some unknown error occurs at this file, skip it
        continue
    gt_path = os.path.join(gt_directory, str(i))
    sm_path = os.path.join(sm_directory, str(i))

    gt_files = os.listdir(gt_path)
    sm_files = os.listdir(sm_path)
    #Now to sort based on their file number. The "key" parameter in sorted is a function based on which the sorting will happen (I use split to exclude the jpg/png from the).
    gt_files_sorted = sorted(gt_files, key = lambda x: int(x.split(".")[0]) )
    sm_files_sorted = sorted(sm_files, key = lambda x: int(x.split(".")[0]) )
    pack = zip(gt_files_sorted, sm_files_sorted)
    print("Files related to video {} sorted.".format(i))

    """
    #Uncomment this segment if you want to debug something, so as to avoid parallel calculations and exit at 5 iterations.
    for n, packed in enumerate(pack):
        a = inner_worker(n, packed, gt_path, sm_path)
        print(a)
        if n==5:
            exit()
    """

    ##https://stackoverflow.com/questions/35663498/how-do-i-return-a-matrix-with-joblib-python
    from joblib import Parallel, delayed
    metric_list = Parallel(n_jobs=8)(delayed(inner_worker)(n, packed, gt_path, sm_path) for n, packed in enumerate(pack)) #run 8 frames simultaneously

    aucj_mean = np.mean([x[0] for x in metric_list])
    aucs_mean = np.mean([x[1] for x in metric_list])
    nss_mean = np.mean([x[2] for x in metric_list])
    cc_mean = np.mean([x[3] for x in metric_list])
    sim_mean = np.mean([x[4] for x in metric_list])

    print("For video number {} the metrics are:".format(i))
    print("AUC-JUDD is {}".format(aucj_mean))
    print("AUC-SHUFFLED is {}".format(aucs_mean))
    print("NSS is {}".format(nss_mean))
    print("CC is {}".format(cc_mean))
    print("SIM is {}".format(sim_mean))
    print("Time elapsed so far: {}".format(datetime.datetime.now().replace(microsecond=0)-start))
    print("==============================")

    final_metric_list.append(( aucj_mean,
                        aucs_mean,
                        nss_mean,
                        cc_mean,
                        sim_mean ))

    with open('metrics.txt', 'wb') as handle:
        pickle.dump(final_metric_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

Aucj = np.mean([y[0] for y in final_metric_list])
Aucs = np.mean([y[1] for y in final_metric_list])
Nss = np.mean([y[2] for y in final_metric_list])
Cc = np.mean([y[3] for y in final_metric_list])
Sim = np.mean([y[4] for y in final_metric_list])

print("Final average of metrics is:")
print("AUC-JUDD is {}".format(Aucj))
print("AUC-SHUFFLED is {}".format(Aucs))
print("NSS is {}".format(Nss))
print("CC is {}".format(Cc))
print("SIM is {}".format(Sim))

