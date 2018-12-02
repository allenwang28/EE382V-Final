from __future__ import print_function, division
import os
import sys
import subprocess

def class_process(dir_path, dst_dir_path, class_name):
  class_path = os.path.join(dir_path, class_name)
  if not os.path.isdir(class_path):
    return

  dst_class_path = os.path.join(dst_dir_path, class_name)
  if not os.path.exists(dst_class_path):
    os.mkdir(dst_class_path)

  for file_name in os.listdir(class_path):
    if '.mp4' not in file_name:
      continue
    name, ext = os.path.splitext(file_name)
    dst_path = os.path.join(dst_class_path, name + '.wav')

    video_file_path = os.path.join(class_path, file_name)
    cmd = 'ffmpeg -i \"{}\" -ab 160k -ac 2 -ar 44100 -vn \"{}\"'.format(video_file_path, dst_path)
    print(cmd)
    subprocess.call(cmd, shell=True)
    print('\n')

if __name__=="__main__":
  dir_path = sys.argv[1]
  dst_dir_path = sys.argv[2]

  for class_name in os.listdir(dir_path):
    class_process(dir_path, dst_dir_path, class_name)

  class_name = 'test'
  class_process(dir_path, dst_dir_path, class_name)
