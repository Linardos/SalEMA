"""
I changed the format when I was initially working so I will have to change it back before sending it
"""
import tarfile
import os

src = "/imatge/lpanagiotis/work/DHF1K/SGmid_predictions"
dst = "/imatge/lpanagiotis/work/DHF1K/SalGANplus"
if not os.path.exists(dst):
    os.mkdir(dst)
format_prototype = "/imatge/lpanagiotis/projects/saliency/dhf1k/annotation"

folders_tomake = os.listdir(format_prototype)

def main():

    for folder in folders_tomake:
        source = os.path.join(src, folder) # new name

        if not os.path.exists(source):
            prev_name = os.path.join(src, str(int(folder)))
            os.rename(prev_name, source) #changing this format naming discrepancy to avoid issues
            print("Renamed {} to {}".format(prev_name, source))

        files = os.listdir(source)
        check = 0

        for file in files:
            prev_file_name = os.path.join(source, file)
            new_file_name = os.path.join(source, file.zfill(8)) #fill zeroes until string has a total of 8 characters
            if check == 0:
                check+=1
                print("Renaming files to match format of authors, starting by changing {} to {}".format(prev_file_name, new_file_name))
            os.rename(prev_file_name, new_file_name)

        tar_file = os.path.join(dst, folder+".tar")

        print("Copying {} to {}..".format(source, tar_file))
        make_tarfile(tar_file, source)

def make_tarfile(tar_file, source_file):
    with tarfile.open(tar_file, "w:gz") as tar:
        tar.add(source_file)

if __name__ == "__main__":
    main()
