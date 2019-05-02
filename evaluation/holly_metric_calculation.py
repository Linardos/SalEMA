from salience_metrics import AUC_Judd, AUC_shuffled, CC, NSS, SIM
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

#===Hollywood has another folder structure
HOLLY_DIR = "/imatge/lpanagiotis/work/UCF-sports/testing"
HOLLY_DIR = "/home/mondestrasz/Documents/Datasets/UCF-sports/testing"
HOLLY_DIR = "/home/linardos/datasets/Hollywood-2/testing/"
MODEL = "SalGANmid_HU"
MODEL = "SalEMA30Afinal_H"

final_metric_list = []

SAVE_METRICS = True
RESCALE_GTs = False
print("Now evaluating on {} predictions".format(MODEL))
continue_calculations = False


def sAUC_sampler(video, M=10):
    # A sampler for the shuffled AUC metric. From the ground truths sample images at random; then aggregate them to feed into sAUC.

    videos = os.listdir(HOLLY_DIR)
    videos.remove(video) # Remove the video being evaluated.
    video_samples = np.random.choice(videos, M, replace=False)

    for i in range(M):
        video_sample = os.path.join(HOLLY_DIR, str(video_samples[i]), "fixation")
        frames = os.listdir(video_sample)
        frames.remove("maps") #matlab files
        frame_sample = np.random.choice(frames, 1, replace=False)[0]

        sample_img = os.path.join(video_sample, frame_sample)
        sample_img = np.asarray(Image.open(sample_img), dtype=np.float32)

        #apparently in Hollywood, images are not of the same size
        tested_video = os.path.join(HOLLY_DIR, video, "fixation")
        tested_video_frames = os.listdir(tested_video)
        tested_video_frames.remove("maps")
        tested_video_frame = np.asarray(Image.open(os.path.join(tested_video, tested_video_frames[0])), dtype=np.float32) #Just need the size doesnt matter what we open.

        sample_img = cv2.resize(sample_img, (tested_video_frame.shape[1], tested_video_frame.shape[0]), interpolation=cv2.INTER_AREA)
        if i == 0:
            avg_img = sample_img
        else:
            avg_img = avg_img + sample_img

    avg_img = np.asarray(avg_img/M, dtype=np.uint8)
    #avg_img = ((avg_img-avg_img.min())/(avg_img.max()-avg_img.min()))*255 #Rescale

    return avg_img

# The directories are named 1-1000 so it should be easy to iterate over them
def inner_worker(n, sAUC_extramap, packed, gt_path, fix_path, sm_path): #packed should be a list of tuples (annotation, prediction)

    gt, fix, sm = packed
    mground_truth = cv2.imread(os.path.join(gt_path, gt),cv2.IMREAD_GRAYSCALE)
    fground_truth = cv2.imread(os.path.join(fix_path, fix),cv2.IMREAD_GRAYSCALE)
    saliency_map = cv2.imread(os.path.join(sm_path, sm),cv2.IMREAD_GRAYSCALE)

    #print("it made it this far")
    if RESCALE_GTs:
        # Avoid doing this in the future. GTs should not be manipulated.
        #print(np.max(mground_truth))
        mground_truth = cv2.resize(mground_truth, (saliency_map.shape[1], saliency_map.shape[0]), interpolation=cv2.INTER_AREA)

        mground_truth[mground_truth==np.max(mground_truth)]=255
        #mground_truth = (mground_truth-np.min(mground_truth))/(np.max(mground_truth)-np.min(mground_truth))
        #mground_truth = mground_truth*255
    else:
        saliency_map = cv2.resize(saliency_map, (mground_truth.shape[1], mground_truth.shape[0]), interpolation=cv2.INTER_AREA)

    #saliency_map_norm = normalize(saliency_map) # The functions are a bit haphazard. Some have normalization within and some do not.

    """
    AUC_SHUF = auc_shuff(saliency_map_norm, ground_truth, sAUC_extramap = ground_truth)
    return AUC_SHUF
    """

    #print("or this far")


    mground_truth = cv2.resize(mground_truth, (0,0), fx=0.5, fy=0.5)
    fground_truth = cv2.resize(fground_truth, (0,0), fx=0.5, fy=0.5)
    sAUC_extramap = cv2.resize(sAUC_extramap, (0,0), fx=0.5, fy=0.5)
    saliency_map = cv2.resize(saliency_map, (0,0), fx=0.5, fy=0.5)

    mground_truth = mground_truth.astype(np.float32)
    fground_truth = fground_truth.astype(np.float32)
    sAUC_extramap = sAUC_extramap.astype(np.float32)
    saliency_map = saliency_map.astype(np.float32)
    # Calculate metrics
    auc_j = AUC_Judd(saliency_map, fground_truth)
    Sauc = AUC_shuffled(saliency_map, fground_truth, sAUC_extramap)
    #print("or this far?")
    Nss = NSS(saliency_map, fground_truth)
    Cc = CC(saliency_map, mground_truth)
    sim = SIM(saliency_map, mground_truth)

    return ( auc_j,
             Sauc,
             Nss,
             Cc,
             sim )


vids = os.listdir(HOLLY_DIR)
start = datetime.datetime.now().replace(microsecond=0)
for i, vid in enumerate(vids):


    gt_path = os.path.join(HOLLY_DIR, vid, "maps")
    fix_path = os.path.join(HOLLY_DIR, vid, "fixation")
    #if i == 57: #Some unknown error occurs at this file, skip it
    #    continue
    sm_path = os.path.join(HOLLY_DIR, vid, "{}_predictions".format(MODEL))

    gt_files = os.listdir(gt_path)
    fix_files = os.listdir(fix_path)
    fix_files.remove("maps") #matlab files
    sm_files = os.listdir(sm_path)

    video_length = len(gt_files)
    #Now to sort based on their file number. The "key" parameter in sorted is a function based on which the sorting will happen (I use split to exclude the jpg/png from the).
    gt_files_sorted = sorted(gt_files)
    fix_files_sorted = sorted(fix_files)
    sm_files_sorted = sorted(sm_files)
    pack = zip(gt_files_sorted, fix_files_sorted, sm_files_sorted)
    print("Files related to video {} sorted.".format(fix_path))

    sAUC_extramap = sAUC_sampler(video=vid)
    #Uncomment this segment if you want to debug something, so as to avoid parallel calculations and exit at 5 iterations.
    """
    for n, packed in enumerate(pack):
        a = inner_worker(n, sAUC_extramap, packed=packed, gt_path=gt_path, fix_path = fix_path, sm_path=sm_path)
        print(a)
        if n==5:
            exit()
    """

    ##https://stackoverflow.com/questions/35663498/how-do-i-return-a-matrix-with-joblib-python
    from joblib import Parallel, delayed
    metric_list = Parallel(n_jobs=4)(delayed(inner_worker)(n, sAUC_extramap , packed=packed, gt_path=gt_path, fix_path = fix_path, sm_path=sm_path) for n, packed in enumerate(pack)) #run 8 frames simultaneously

    print("Final average of metrics is:")
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

if SAVE_METRICS:
    np.save("{}_metrics".format(MODEL),final_metric_list)

Aucj = np.mean([y[0] for y in final_metric_list])
Aucs = np.mean([y[1] for y in final_metric_list])
Nss = np.mean([y[2] for y in final_metric_list])
Cc = np.mean([y[3] for y in final_metric_list])
Sim = np.mean([y[4] for y in final_metric_list])

print("Evaluation on {} finished.".format(MODEL))
print("Final average of metrics is:")
print("AUC-JUDD is {}".format(Aucj))
print("AUC-SHUFFLED is {}".format(Aucs))
print("NSS is {}".format(Nss))
print("CC is {}".format(Cc))
print("SIM is {}".format(Sim))
