import torch
import os
import datetime
import numpy as np
from torch.utils import data
from PIL import Image
from torchvision import utils


# The DataLoader for our specific video datataset with extracted frames
class DHF1K_frames(data.Dataset):

  def __init__(self, split, clip_length, number_of_videos, frames_path = "/imatge/lpanagiotis/work/DHF1K/frames", gt_path = "/imatge/lpanagiotis/work/DHF1K/maps",  val_perc = 0.15, transforms = None):

        self.transforms = transforms
        self.cl = clip_length
        self.frames_path = frames_path # in our case it's salgan saliency maps
        self.gt_path = gt_path#ground truth


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

          # Load data and get ground truth
          path_to_frame = os.path.join(self.frames_path, str(true_index), frame)

          X = Image.open(path_to_frame)
          if self.transforms:
            X = self.transforms(X)
          X = (X - X.min())/(X.max()-X.min()) # Normalize min 0 max 1

          path_to_gt = os.path.join(self.gt_path, str(true_index), gts[frame])

          y = Image.open(path_to_gt)
          if self.transforms:
            y = self.transforms(y)
          y = (y - y.min())/(y.max()-y.min())

          data.append(X.unsqueeze(0))
          gt.append(y.unsqueeze(0))

          if (i+1)%self.cl == 0 or i == (len(frames)-1):
            #print(np.array(data).shape) #looks okay

            data_tensor = torch.cat(data,0) #bug was actually here
            gt_tensor = torch.cat(gt,0)
            packed.append((data_tensor,gt_tensor)) # pack a list of data with the corresponding list of ground truths
            data = []
            gt = []


        return packed





# DataLoader for inference Works for EgoMon and GTEA
class Ego_frames(data.Dataset):

  def __init__(self, clip_length, frames_path, transforms = None, num_videos=None):

        self.transforms = transforms
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
            if num_videos!=None:
              if i>num_videos:
                break


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

          X = Image.open(path_to_frame)
          if self.transforms:
            X = self.transforms(X)
          X = (X - X.min())/(X.max()-X.min()) # Normalize min 0 max 1

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



# DataLoader for inference
class EpicKitchen_frames(data.Dataset):

  def __init__(self, clip_length, frames_path, transforms = None):

        self.transforms = transforms
        self.cl = clip_length
        self.frames_path = frames_path # in our case it's salgan saliency maps

        self.video_dict = {}

        start = datetime.datetime.now().replace(microsecond=0) # Gives accurate human readable time, rounded down not to include too many decimals
        activities_folders = os.listdir(frames_path)
        self.match_i_to_vid_path = {}
        count = 0
        for x in ["train", "test"]:
          root_path = os.path.join(frames_path, x)
          people = os.listdir(root_path)
          for person in people:
            person_path = os.path.join(frames_path, x, person)
            videos = os.listdir(person_path)

            for video in videos: #Every video is a directory
                # Define our source file
                source_video = os.path.join(person_path, video)
                # Define our destination directory
                self.match_i_to_vid_path[count] = source_video

                frame_files = os.listdir(source_video)
                frame_files_sorted = sorted(frame_files)
                self.video_dict[count]=frame_files_sorted


                count+=1

            print("Frames for {} organized.".format(person))
            print("Time elapsed so far: {}".format(datetime.datetime.now().replace(microsecond=0)-start))


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.video_dict.keys())

  def __getitem__(self, video_index):

        'Generates one sample of data'
        src_vid = self.match_i_to_vid_path[video_index]
        frames = self.video_dict[video_index]
        # Select sample video (frame list), in our case saliency map list
        data = []
        frame_names = []
        packed = []
        for i, frame in enumerate(frames):
          # Load data
          path_to_frame = os.path.join(src_vid, frame)

          X = Image.open(path_to_frame).convert("L")
          if self.transforms:
            X = self.transforms(X)
          X = (X - X.min())/(X.max()-X.min()) # Normalize min 0 max 1

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
