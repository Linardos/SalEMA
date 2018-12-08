import torch
import os
import datetime
import numpy as np
import cv2
from torch.utils import data
from torchvision import utils


# The DataLoader for our specific video datataset with extracted frames
class DHF1K_frames(data.Dataset):

  def __init__(self, split, clip_length, number_of_videos, resolution=None, frames_path = "/imatge/lpanagiotis/work/DHF1K/frames", gt_path = "/imatge/lpanagiotis/work/DHF1K/maps",  val_perc = 0.15):

        self.cl = clip_length
        self.frames_path = frames_path # in our case it's salgan saliency maps
        self.gt_path = gt_path#ground truth
        self.ImageNet_mean = [103.939, 116.779, 123.68]
        self.resolution = resolution
        # A list to keep all video lists of salgan predictions, which will be our dataset.
        self.video_list = []

        # A list to keep all the dictionaries of ground truth - saliency map pairings for each video
        self.gts_list = []

        start = datetime.datetime.now().replace(microsecond=0) # Gives accurate human readable time, rounded down not to include too many decimals
        for i in range(1, number_of_videos+1): #700 videos in DHF1K

            # The way the folder structure is organized allows to simply iterate over the range of the number of total videos.
            gt_files = os.listdir(os.path.join(self.gt_path,str(i)))
            frame_files = os.listdir(os.path.join(self.frames_path,str(i)))
            #print("for video {} the frames are {}".format(i, len(frame_files))) # This is correct

            # Now to sort based on their file number. The "key" parameter in sorted is a function based on which the sorting will happen (I use split to exclude the jpg/png from the).
            gt_files_sorted = sorted(gt_files, key = lambda x: int(x.split(".")[0]) )
            frame_files_sorted = sorted(frame_files, key = lambda x: int(x.split(".")[0]) )
            pack = zip(gt_files_sorted, frame_files_sorted)

            # a list of lists
            self.video_list.append(frame_files_sorted)

            # Make dictionary where keys are the saliency maps and values are the ground truths
            gt_frame_pairings = {}
            for gt, frame in pack:
                gt_frame_pairings[frame] = gt

            self.gts_list.append(gt_frame_pairings)
            if i%50==0:
                print("Pairings related to video {} organized.".format(i))
                print("Time elapsed so far: {}".format(datetime.datetime.now().replace(microsecond=0)-start))



        # pack a list of data with the corresponding list of ground truths
        # Split the dataset to validation and training
        limit = int(round(val_perc*len(self.video_list)))
        if split == "validation":
          self.video_list = self.video_list[:limit]
          self.gts_list = self.gts_list[:limit]
          self.first_video_no = 1 #This needs to be specified to find the correct directory in our case. It will be different for each split since these directories signify videos.
        elif split == "train":
          self.video_list = self.video_list[limit:]
          self.gts_list = self.gts_list[limit:]
          self.first_video_no = limit+1
        elif split == None:
          self.first_video_no = 1




  def __len__(self):
        'Denotes the total number of samples'
        return len(self.video_list)

  def __getitem__(self, video_index):

        'Generates one sample of data'
        # Select sample video (frame list), in our case saliency map list
        frames = self.video_list[video_index]
        gts = self.gts_list[video_index]

        # Due to the split in train and validation we need to add this number to the video_index to find the correct video (to match the files in the path with the video list the training part uses)
        true_index = self.first_video_no + video_index #this matches the correct video number

        data = []
        gt = []
        packed = []
        for i, frame in enumerate(frames):

          # Load and preprocess frames
          path_to_frame = os.path.join(self.frames_path, str(true_index), frame)

          X = cv2.imread(path_to_frame)
          if self.resolution!=None:
            X = cv2.resize(X, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
          X = X.astype(np.float32)
          X -= self.ImageNet_mean
          #X = (X-np.min(X))/(np.max(X)-np.min(X))
          X = torch.FloatTensor(X)
          X = X.permute(2,0,1) # swap channel dimensions

          # Load and preprocess ground truth (saliency maps)
          path_to_gt = os.path.join(self.gt_path, str(true_index), gts[frame])

          y = cv2.imread(path_to_gt, 0) # Load as grayscale
          if self.resolution!=None:
            y = cv2.resize(y, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
          y = (y-np.min(y))/(np.max(y)-np.min(y))
          y = torch.FloatTensor(y)

          data.append(X.unsqueeze(0))
          gt.append(y.unsqueeze(0))

          if (i+1)%self.cl == 0 or i == (len(frames)-1):

            data_tensor = torch.cat(data,0)
            gt_tensor = torch.cat(gt,0)
            packed.append((data_tensor,gt_tensor)) # pack a list of data with the corresponding list of ground truths
            data = []
            gt = []


        return packed
