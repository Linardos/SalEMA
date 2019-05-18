import torch
import os
import datetime
import numpy as np
import cv2
from torch.utils import data
from torchvision import utils


# The DataLoader for our specific video datataset with extracted frames
class DHF1K_frames(data.Dataset):

  def __init__(self, split, clip_length, number_of_videos, starting_video, root_path, load_gt, resolution=None, val_perc = 0.01):

        self.starting_video = starting_video
        self.cl = clip_length
        self.frames_path = os.path.join(root_path, "frames") # in our case it's salgan saliency maps
        self.load_gt = load_gt
        if load_gt:
          self.gt_path = os.path.join(root_path, "maps")#ground truth
        self.ImageNet_mean = [103.939, 116.779, 123.68]
        self.resolution = resolution
        # A list to keep all video lists of salgan predictions, which will be our dataset.
        self.video_list = []

        # A list to keep all the dictionaries of ground truth - saliency map pairings for each video
        self.gts_list = []

        start = datetime.datetime.now().replace(microsecond=0) # Gives accurate human readable time, rounded down not to include too many decimals
        for i in range(starting_video, number_of_videos+1): #700 videos in DHF1K

            # The way the folder structure is organized allows to simply iterate over the range of the number of total videos.
            frame_files = os.listdir(os.path.join(self.frames_path,str(i)))
            #print("for video {} the frames are {}".format(i, len(frame_files))) # This is correct

            # Now to sort based on their file number. The "key" parameter in sorted is a function based on which the sorting will happen (I use split to exclude the jpg/png from the).
            frame_files_sorted = sorted(frame_files, key = lambda x: int(x.split(".")[0]) )
            # a list of lists
            self.video_list.append(frame_files_sorted)

            if load_gt:
              gt_files = os.listdir(os.path.join(self.gt_path,str(i)))
              gt_files_sorted = sorted(gt_files, key = lambda x: int(x.split(".")[0]) )
              pack = zip(gt_files_sorted, frame_files_sorted)

              # Make dictionary where keys are the saliency maps and values are the ground truths
              gt_frame_pairings = {}
              for gt, frame in pack:
                  gt_frame_pairings[frame] = gt

              self.gts_list.append(gt_frame_pairings)

            if i%50==0:
              print("Video {} finished.".format(i))
              print("Time elapsed so far: {}".format(datetime.datetime.now().replace(microsecond=0)-start))

        # pack a list of data with the corresponding list of ground truths
        # Split the dataset to validation and training
        limit = int(round(val_perc*len(self.video_list)))
        if split == "validation":
          self.video_list = self.video_list[:limit]
          self.gts_list = self.gts_list[:limit]
          self.first_video_no = starting_video #This needs to be specified to find the correct directory in our case. It will be different for each split since these directories signify videos.
        elif split == "train":
          self.video_list = self.video_list[limit:]
          self.gts_list = self.gts_list[limit:]
          self.first_video_no = limit+starting_video
        elif split == None:
          self.first_video_no = starting_video




  def __len__(self):
        'Denotes the total number of samples'
        return len(self.video_list)

  def __getitem__(self, video_index):

        'Generates one sample of data'
        # Select sample video (frame list), in our case saliency map list
        frames = self.video_list[video_index]
        if self.load_gt:
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

          data.append(X.unsqueeze(0))
          # Load and preprocess ground truth (saliency maps)
          if self.load_gt:

            path_to_gt = os.path.join(self.gt_path, str(true_index), gts[frame])

            y = cv2.imread(path_to_gt, 0) # Load as grayscale
            if self.resolution!=None:
              y = cv2.resize(y, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
            y = (y-np.min(y))/(np.max(y)-np.min(y))
            y = torch.FloatTensor(y)

            gt.append(y.unsqueeze(0))

          if (i+1)%self.cl == 0 or i == (len(frames)-1):

            data_tensor = torch.cat(data,0)
            data = []
            if self.load_gt:
              gt_tensor = torch.cat(gt,0)
              gt = []
              packed.append((data_tensor,gt_tensor)) # pack a list of data with the corresponding list of ground truths
            else:
              packed.append((data_tensor, "_"))


        return packed

# The DataLoader for our specific video datataset with extracted frames
class Hollywood_frames(data.Dataset):

  def __init__(self, clip_length, resolution=None, root_path = "~/work/Hollywood-2/testing/", load_gt = False):
        """
        Frames should be under a folder "images" and ground truths under folder named "maps"
        """

        self.cl = clip_length
        self.root_path = root_path # in our case it's salgan saliency maps
        self.ImageNet_mean = [103.939, 116.779, 123.68]
        self.resolution = resolution
        self.load_gt = load_gt
        # A list to keep all video lists of salgan predictions, which will be our dataset.
        self.video_list = []

        # A list to keep all the dictionaries of ground truth - saliency map pairings for each video
        self.gts_list = []

        # A list to match an index to the video name
        self.video_name_list = []

        sample_list = os.listdir(root_path)
        start = datetime.datetime.now().replace(microsecond=0) # Gives accurate human readable time, rounded down not to include too many decimals
        count = 0
        for i in sample_list:

            frame_files = [os.path.join(self.root_path, str(i), "images", file) for file in  os.listdir(os.path.join(self.root_path, str(i), "images"))]
            #print("for video {} the frames are {}".format(i, len(frame_files))) # This is correct

            # Now to sort based on their file number. The "key" parameter in sorted is a function based on which the sorting will happen (I use split to exclude the jpg/png from the).
            frame_files_sorted = sorted(frame_files)
            # a list of lists
            self.video_list.append(frame_files_sorted)
            self.video_name_list.append(i)

            count += 1

            if count%50==0:
              print("Video {} (Number {}) finished.".format(i, count))
              print("Time elapsed so far: {}".format(datetime.datetime.now().replace(microsecond=0)-start))


  def video_names(self):
      return self.video_name_list

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.video_list)

  def __getitem__(self, video_index):

        'Generates one sample of data'
        # Select sample video (frame list), in our case saliency map list
        frames = self.video_list[video_index]

        data = []
        gt = []
        packed = []
        #print("Frame: {}".format(frames[0]))
        for i, path_to_frame in enumerate(frames):

          X = cv2.imread(path_to_frame)
          if self.resolution!=None:
            X = cv2.resize(X, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
          X = X.astype(np.float32)
          X -= self.ImageNet_mean
          #X = (X-np.min(X))/(np.max(X)-np.min(X))
          X = torch.FloatTensor(X)
          X = X.permute(2,0,1) # swap channel dimensions

          data.append(X.unsqueeze(0))
          # Load and preprocess ground truth (saliency maps)
          if self.load_gt:
            path_to_gt = path_to_frame.replace("images", "maps")
            y = cv2.imread(path_to_gt, 0) # Load as grayscale
            if self.resolution!=None:
              y = cv2.resize(y, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
            y = (y-np.min(y))/(np.max(y)-np.min(y))
            y = torch.FloatTensor(y)

            gt.append(y.unsqueeze(0))

            """
            print("frame: {}".format(path_to_frame))
            print("gtruth: {}".format(path_to_gt))
            exit()
            """

          if (i+1)%self.cl == 0 or i == (len(frames)-1):

            data_tensor = torch.cat(data,0)
            data = []
            if self.load_gt:
              gt_tensor = torch.cat(gt,0)
              gt = []
              packed.append((data_tensor,gt_tensor)) # pack a list of data with the corresponding list of ground truths
            else:
              packed.append((data_tensor, "_"))


        return packed



# The DataLoader for our specific video datataset with extracted frames
class DAVIS_frames(data.Dataset):

  def __init__(self, clip_length, resolution=None, root_path = "~/projects/segmentation/davis2017/JPEGImages/480p", load_gt = False):
        """
        Frames should be under a folder "images" and ground truths under folder named "maps"
        """

        self.cl = clip_length
        self.root_path = root_path # in our case it's salgan saliency maps
        self.ImageNet_mean = [103.939, 116.779, 123.68]
        self.resolution = resolution
        self.load_gt = load_gt
        # A list to keep all video lists of salgan predictions, which will be our dataset.
        self.video_list = []

        # A list to keep all the dictionaries of ground truth - saliency map pairings for each video
        self.gts_list = []

        # A list to match an index to the video name
        self.video_name_list = []

        sample_list = os.listdir(root_path)
        start = datetime.datetime.now().replace(microsecond=0) # Gives accurate human readable time, rounded down not to include too many decimals
        count = 0
        for i in sample_list:

            frame_files = [os.path.join(self.root_path, str(i), file) for file in os.listdir(os.path.join(self.root_path, str(i)))]
            #print("for video {} the frames are {}".format(i, len(frame_files))) # This is correct

            # Now to sort based on their file number. The "key" parameter in sorted is a function based on which the sorting will happen (I use split to exclude the jpg/png from the).
            frame_files_sorted = sorted(frame_files)
            # a list of lists
            self.video_list.append(frame_files_sorted)
            self.video_name_list.append(i)

            count += 1

            if count%50==0:
              print("Video {} (Number {}) finished.".format(i, count))
              print("Time elapsed so far: {}".format(datetime.datetime.now().replace(microsecond=0)-start))


  def video_names(self):
      return self.video_name_list

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.video_list)

  def __getitem__(self, video_index):

        'Generates one sample of data'
        # Select sample video (frame list), in our case saliency map list
        frames = self.video_list[video_index]

        data = []
        gt = []
        packed = []
        #print("Frame: {}".format(frames[0]))
        for i, path_to_frame in enumerate(frames):

          X = cv2.imread(path_to_frame)
          if self.resolution!=None:
            X = cv2.resize(X, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
          X = X.astype(np.float32)
          X -= self.ImageNet_mean
          #X = (X-np.min(X))/(np.max(X)-np.min(X))
          X = torch.FloatTensor(X)
          X = X.permute(2,0,1) # swap channel dimensions

          data.append(X.unsqueeze(0))
          # Load and preprocess ground truth (saliency maps)
          if self.load_gt:
            path_to_gt = path_to_frame.replace("images", "maps")
            y = cv2.imread(path_to_gt, 0) # Load as grayscale
            if self.resolution!=None:
              y = cv2.resize(y, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
            y = (y-np.min(y))/(np.max(y)-np.min(y))
            y = torch.FloatTensor(y)

            gt.append(y.unsqueeze(0))

            """
            print("frame: {}".format(path_to_frame))
            print("gtruth: {}".format(path_to_gt))
            exit()
            """

          if (i+1)%self.cl == 0 or i == (len(frames)-1):

            data_tensor = torch.cat(data,0)
            data = []
            if self.load_gt:
              gt_tensor = torch.cat(gt,0)
              gt = []
              packed.append((data_tensor,gt_tensor)) # pack a list of data with the corresponding list of ground truths
            else:
              packed.append((data_tensor, "_"))


        return packed




# DataLoader for inference Works for EgoMon and GTEA
class Ego_frames(data.Dataset):

  def __init__(self, clip_length, frames_path = "/imatge/lpanagiotis/work/Egomon/frames", resolution = None):

        self.ImageNet_mean = [103.939, 116.779, 123.68]
        self.resolution = resolution
        self.cl = clip_length
        self.frames_path = frames_path # in our case it's salgan saliency maps

        self.video_dict = {}

        start = datetime.datetime.now().replace(microsecond=0) # Gives accurate human readable time, rounded down not to include too many decimals
        activities_folders = os.listdir(frames_path)
        self.match_i_to_act = {}
        for i, activity in enumerate(activities_folders):
            self.match_i_to_act[i] = activity

            complete_path = os.path.join(frames_path, activity)
            frame_files = os.listdir(complete_path)

            frame_files_sorted = sorted(frame_files)
            #print(frame_files_sorted[0:30]) looks good
            # a list of lists
            self.video_dict[activity]=frame_files_sorted

            print("Frames for {} organized.".format(activity))
            print("Time elapsed so far: {}".format(datetime.datetime.now().replace(microsecond=0)-start))


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.video_dict.keys())

  def __getitem__(self, video_index):

        activity = self.match_i_to_act[video_index]
        'Generates one sample of data'
        # Select sample video (frame list), in our case saliency map list
        frames = self.video_dict[activity]

        data = []
        frame_names = []
        packed = []
        for i, frame in enumerate(frames):
          # Load data
          path_to_frame = os.path.join(self.frames_path, activity, frame)

          X = cv2.imread(path_to_frame)
          if self.resolution!=None:
            X = cv2.resize(X, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
          X = X.astype(np.float32)
          X -= self.ImageNet_mean
          #X = (X-np.min(X))/(np.max(X)-np.min(X))
          X = torch.FloatTensor(X)
          X = X.permute(2,0,1) # swap channel dimensions

          data.append(X.unsqueeze(0))
          frame_names.append(frame)

          if (i+1)%self.cl == 0 or i == (len(frames)-1):
            #print(np.array(data).shape) #looks okay

            data_tensor = torch.cat(data,0) #bug was actually here
            packed.append((frame_names, data_tensor))
            data = []
            frame_names = []

        #print("Maybe inside here")

        return packed
