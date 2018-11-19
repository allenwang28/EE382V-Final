import os
import sys

def class_process(dir_path, class_name, dest_path):
    pass

if __name__ == "__main__":
    dir_path = sys.argv[1]
    dest_path = sys.argv[2]

    for class_name in os.listdir(dir_path):
        class_process(dir_path, class_name, dest_path)

