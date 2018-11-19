import os
import sys
import shutil

def class_process(dir_path, dst_dir_path, class_name):
    class_path = os.path.join(dir_path, class_name)
    if not os.path.isdir(class_path):
        return

    dst_class_path = os.path.join(dst_dir_path, class_name)
    if not os.path.exists(dst_class_path):
        os.mkdir(dst_class_path)

    for file_name in os.listdir(class_path):
        if '.wav' not in file_name:
            continue
        name, ext = os.path.splitext(file_name)

        dst_path = os.path.join(dst_class_path, name, name + ext)

        print ("Copying {0} to {1}".format(file_name, dst_path))
        # shutil.copy2(file_name, dst_path)
                

if __name__ == "__main__":
    dir_path = sys.argv[1]
    dst_dir_path = sys.argv[2]

    for class_name in os.listdir(dir_path):
        class_process(dir_path, dst_dir_path, class_name)

