import os
import sys
import subprocess

if __name__ == "__main__":
    dir_path = sys.argv[1]
    dst_dir_path = sys.argv[2]

    for file_name in os.listdir(dir_path):
        if '.mp4' not in file_name:
            continue
        name, ext = os.path.splitext(file_name)
        dst_path = os.path.join(dst_dir_path, name, '.wav')

        video_file_path = os.path.join(dir_path, file_name)
        try:
            if os.path.exists(dst_directory_path):
                if not os.path.exists(os.path.join(dst_directory_path, 'image_00001.jpg')):
                subprocess.call('rm -r {}'.format(dst_directory_path), shell=True)
                print('remove {}'.format(dst_directory_path))
                os.mkdir(dst_directory_path)
            else:
                continue
            else:
                os.mkdir(dst_directory_path)
        except:
            print(dst_directory_path)
            continue
        cmd = 'ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn {}/image_%05d.jpg'.format(video_file_path, dst_path)
        print(cmd)
        subprocess.call(cmd, shell=True)
        print('\n')
