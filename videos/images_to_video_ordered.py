from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os

work_dir = "/imatge/lpanagiotis/work/DHF1K/"
#path to image folder
sm_subdir = "SGplus_predictions"
sm_dir = os.path.join(work_dir, sm_subdir)
gt_dir = os.path.join(work_dir, "maps")
data_dir = os.path.join(work_dir, "frames")


output_dir = "./DHF1K_videos"
video_nums = [2, 5, 10, 50, 100, 300, 600]

#function that makes the video - change fps if needed
def make_video(src, images, output, fps=6, size=None,
               is_color=True, format="XVID"):
    """
    Create a video from a list of images.

    @param      output      output video name, extension is added afterwards
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        image = os.path.join(src, image)
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter("{}.avi".format(output), fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()

def output_video(src, dst, num):
    if not os.path.exists(dst):
        os.makedirs(dst)
    else:
        print("Be warned, you are about to write on an existing folder {}. If this is not intentional cancel now.".format(dst))

    image_dir = os.path.join(src, str(num))
    video_dir = os.path.join(dst, str(num))
    #get images with random order
    images = os.listdir(image_dir)
    images_ordered = sorted(images, key = lambda x: int(x.split(".")[0]) )

    #call the function - change fps if needed
    make_video(image_dir, images_ordered, output = video_dir, fps=6, size=None,
           is_color=True, format="XVID")

if __name__ == "__main__":

    for num in video_nums:
        print("Now accumulating frames into videos for sample number {}".format(num))
        output_video(src=data_dir, dst=os.path.join(output_dir, "data"), num=num)
        output_video(src=gt_dir, dst=os.path.join(output_dir, "ground_truth"), num=num)
        output_video(src=sm_dir, dst=os.path.join(output_dir, sm_subdir), num=num)
        print("Videos related to sample number {} created".format(num))
