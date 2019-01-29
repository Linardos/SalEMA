import shutil
import os

DHF1K_gt_root = "/imatge/lpanagiotis/projects/saliency/dhf1k/annotation"

vids = sorted(os.listdir(DHF1K_gt_root))

for vid in vids:
    src_fixations_path = os.path.join(DHF1K_gt_root, vid, "fixation")
    dst_fixations_path = os.path.join("/imatge/lpanagiotis/work/DHF1K/fixations", str(int(vid))) # originally I used integers and not "0001" strings. the latter might be preferable but it's better to be consistent, so.
    print("Copying fixations from {} to {}...".format(src_fixations_path, dst_fixations_path))
    shutil.copytree(src_fixations_path, dst_fixations_path, ignore="maps")
