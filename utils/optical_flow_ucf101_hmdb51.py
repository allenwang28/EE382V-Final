import os
import cv2
import sys

def class_process_of(dir_path, dst_dir_path, class_name):
    class_path = os.path.join(dir_path, class_name)
    if not os.path.isdir(class_path):
        return

    dst_class_path = os.path.join(dst_dir_path, class_name)

if __name__ == "__main__":
    dir_path = sys.argv[1]
    dst_dir_path = sys.argv[2]

    for class_name in os.listdir(dir_path):
        class_process_of_(dir_path, dst_dir_path, class_name)
