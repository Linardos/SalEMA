#https://www.pythoncentral.io/how-to-recursively-copy-a-directory-folder-in-python/

from shutil import copytree
import os

ego_activities = os.listdir("/imatge/lpanagiotis/projects/saliency/public_html/2016-egomon/video_clean/")

for a in ego_activities:
    src = "/imatge/lpanagiotis/projects/saliency/public_html/2016-egomon/video_clean/{}/frames_{}".format(a,a)
    dst = "/imatge/lpanagiotis/work/Egomon/frames/{}".format(a)

    copytree(src, dst)
