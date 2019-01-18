from salience_metrics import auc_judd, auc_shuff, cc, nss, similarity, normalize_map
"""
DHF1K paper: "we  employ  five  classic  met-rics,  namely  Normalized  Scanpath  Saliency  (NSS),  Sim-ilarity Metric (SIM), Linear Correlation Coefficient (CC),AUC-Judd (AUC-J), and shuffled AUC (s-AUC).""
"""
import cv2
import os
import numpy as np
import pickle
import datetime
import torch
from PIL import Image

gt_directory = "/imatge/lpanagiotis/work/Egomon/maps"
#sm_directory = "/imatge/lpanagiotis/work/Egomon/SGplus_predictions"
#sm_directory = "/imatge/lpanagiotis/work/Egomon/SG_predictions"
#sm_directory = "/imatge/lpanagiotis/work/Egomon/SGtuned_predictions"
#sm_directory = "/imatge/lpanagiotis/work/Egomon/SGplus_predictions_J"
sm_directory = "/imatge/lpanagiotis/work/Egomon/SGmid_predictions" # This is with JJ weights
DHF1K_directory = "/imatge/lpanagiotis/work/DHF1K/maps"
RESCALE_GTs = True # It seems rescaling the saliency map completely screws the results, whereas scaling the ground truths doesnt. In the DHF1K we saw that there is no significant discrepancy between the results if you do this.
print("Now evaluating on {}".format(sm_directory))
continue_calculations = False

NUMBER_OF_VIDEOS = 7

if continue_calculations:
    with open('metrics.txt', 'rb') as handle:
        final_metric_list = pickle.load(handle)
else:
    final_metric_list = []


def sAUC_sampler(M=50, DHF1K_directory=DHF1K_directory):
    # A sampler for the shuffled AUC metric. From the ground truths sample images at random; then aggregate them to feed into sAUC.

    videos = list(range(1,701))
    video_sample_inds = np.random.choice(videos, M, replace=False)

    for i in range(M):

        video_sample = os.path.join(DHF1K_directory, str(video_sample_inds[i]))
        frames = os.listdir(video_sample)
        frame_sample = np.random.choice(frames, 1, replace=False)[0]

        sample_img = os.path.join(video_sample, frame_sample)
        sample_img = np.asarray(Image.open(sample_img), dtype=np.float32)

        if i == 0:
            avg_img = sample_img
        else:
            avg_img = avg_img + sample_img

    avg_img = np.asarray(avg_img/M, dtype=np.uint8)
    #avg_img = ((avg_img-avg_img.min())/(avg_img.max()-avg_img.min()))*255 #Rescale

    return avg_img

# The directories are named 1-1000 so it should be easy to iterate over them
def inner_worker(n, sAUC_extramap, packed, gt_path, sm_path): #packed should be a list of tuples (annotation, prediction)

    gt, sm = packed
    ground_truth = cv2.imread(os.path.join(gt_path, gt),cv2.IMREAD_GRAYSCALE)
    saliency_map = cv2.imread(os.path.join(sm_path, sm),cv2.IMREAD_GRAYSCALE)


    if RESCALE_GTs:
        # Avoid doing this in the future. GTs should not be manipulated.
        #print(np.max(ground_truth))
        ground_truth = cv2.resize(ground_truth, (saliency_map.shape[1], saliency_map.shape[0]), interpolation=cv2.INTER_AREA)
        # some error seems to occur, particularly on the first 3 metrics after the resize. CC and SIM are calculated normally. Noticeably the maximum value of 255 changes for the ground truth. It makes sense that this confuses location based metrics that aim to compare fixation points (hence maximum value locations).

        ground_truth[ground_truth==np.max(ground_truth)]=255
        # Rescaling turned out to cause a mess to the distribution, which results in a mess for the distribution based metrics (CC , SIM). This is evident from the fact that before rescaling the mean is close to 9 but afterwards it's close to 0. It is apparent that rescaling saliency maps down and then up again causes issues and should be avoided.
        #ground_truth = (ground_truth-np.min(ground_truth))/(np.max(ground_truth)-np.min(ground_truth))
        #ground_truth = ground_truth*255
    else:
        saliency_map = cv2.resize(saliency_map, (ground_truth.shape[1], ground_truth.shape[0]), interpolation=cv2.INTER_LINEAR)

    saliency_map_norm = normalize_map(saliency_map) # The functions are a bit haphazard. Some have normalization within and some do not.
    ground_truth_norm = normalize_map(ground_truth)
    #print(np.where(saliency_map_norm==1))
    #print(np.where(ground_truth_norm==1))
    """
    AUC_SHUF = auc_shuff(saliency_map_norm, ground_truth, other_map = ground_truth)
    return AUC_SHUF
    """
    # Calculate metrics
    AUC_JUDD = auc_judd(saliency_map_norm, ground_truth)
    AUC_SHUF = auc_shuff(saliency_map_norm, ground_truth, sAUC_extramap)
    # the other ones have normalization within:
    NSS = nss(saliency_map, ground_truth)

    return ( AUC_JUDD,
             AUC_SHUF,
             NSS )

start = datetime.datetime.now().replace(microsecond=0)
for i in range(0, NUMBER_OF_VIDEOS):

    activities = os.listdir(gt_directory)
    if len(activities) < NUMBER_OF_VIDEOS:
        "The source directory {} includes less videos than expected".format(gt_directory)
        exit()

    gt_path = os.path.join(gt_directory, activities[i])
    sm_path = os.path.join(sm_directory, activities[i])

    video_tag = activities[i]

    gt_files = os.listdir(gt_path)
    sm_files = os.listdir(sm_path)

    video_length = len(gt_files)
    #Now to sort based on their file number. The "key" parameter in sorted is a function based on which the sorting will happen (I use split to exclude the jpg/png from the). # .split(".")[0].split("_")[-1] will find the number in a picture of the following format : 'a_b_123.jpg' => '123'
    gt_files_sorted = sorted(gt_files, key = lambda x: int(x.split(".")[0].split("_")[-1]) )
    sm_files_sorted = sorted(sm_files, key = lambda x: int(x.split(".")[0].split("_")[-1]) )
    pack = zip(gt_files_sorted, sm_files_sorted)
    print("Files related to video {} sorted.".format(i))

    sAUC_extramap = sAUC_sampler()
    #Uncomment this segment if you want to debug something, so as to avoid parallel calculations and exit at 5 iterations.
    """
    for n, packed in enumerate(pack):
        a = inner_worker(n, sAUC_extramap, packed=packed, gt_path=gt_path, sm_path=sm_path)
        print(a)
        if n==15:
            exit()
    """

    ##https://stackoverflow.com/questions/35663498/how-do-i-return-a-matrix-with-joblib-python
    from joblib import Parallel, delayed
    metric_list = Parallel(n_jobs=8)(delayed(inner_worker)(n, sAUC_extramap , packed=packed, gt_path=gt_path, sm_path=sm_path) for n, packed in enumerate(pack)) #run 8 frames simultaneously

    aucj_mean = np.mean([x[0] for x in metric_list])
    aucs_mean = np.mean([x[1] for x in metric_list])
    nss_mean = np.mean([x[2] for x in metric_list])

    print("For video {} the metrics are:".format(activities[i]))
    print("AUC-JUDD is {}".format(aucj_mean))
    print("AUC-SHUFFLED is {}".format(aucs_mean))
    print("NSS is {}".format(nss_mean))
    print("Time elapsed so far: {}".format(datetime.datetime.now().replace(microsecond=0)-start))
    print("==============================")

    final_metric_list.append(( aucj_mean,
                        aucs_mean,
                        nss_mean))

    with open('metrics.txt', 'wb') as handle:
        pickle.dump(final_metric_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

Aucj = np.mean([y[0] for y in final_metric_list])
Aucs = np.mean([y[1] for y in final_metric_list])
Nss = np.mean([y[2] for y in final_metric_list])

print("Evaluation on directory {} finished.".format(sm_directory))
print("Final average of metrics is:")
print("AUC-JUDD is {}".format(Aucj))
print("AUC-SHUFFLED is {}".format(Aucs))
print("NSS is {}".format(Nss))
