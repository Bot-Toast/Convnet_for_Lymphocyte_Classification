import glob
import os
import random
import shutil


# Iterates through a chosen file directory renaming all the files to user specification.
def rename_files(cat, f_ext, file_path):
    os.chdir('E:\\Dissertation\\Dissertation_code\\find_the_bad_guys\\Datasets\\Train\\Normal lymphomcytes')
    print(os.getcwd())

    for i, j in enumerate(os.listdir()):
        new_name = f'{cat}{i}.{f_ext}'
        os.rename(j, new_name)


# to use this properly the sets must not be premixed. There is definitely a more elegant way of doing this.
# Currently, hardcoded for ease of use.

def rand_valid_select(in_path, cat, out_path):
    os.chdir('E:\\Dissertation\\Dissertation_code\\find_the_bad_guys\\Datasets')
    for i in random.sample(glob.glob(f'{in_path}\\{cat}*.jpg'), 44):
        shutil.move(i, f'{out_path}\\{cat}')


in_path = 'E:\\Dissertation\\Dissertation_code\\find_the_bad_guys\\Datasets\\train\\big\\'
out_path = 'E:\\Dissertation\\Dissertation_code\\find_the_bad_guys\\Datasets\\Validation\\'
dir_str = input("enter a directory to move images from: ")
klass = dir_str

print(klass)
print(in_path)
print(out_path)

rand_valid_select(in_path, klass, out_path)
