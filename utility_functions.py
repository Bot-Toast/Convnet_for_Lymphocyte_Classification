import glob
import os
import random
import shutil

# don't call these within main, create a separate project!


# Iterates through a chosen file directory renaming all the files to user specification.
# rationale: To easily distribute large datasets into different collections, using the rand_valid_select function.
def rename_files(cat, f_ext, file_path):
    os.chdir('E:\\Dissertation\\Dissertation_code\\find_the_bad_guys\\Datasets\\Train\\Normal lymphomcytes')
    print(os.getcwd())

    for i, j in enumerate(os.listdir()):
        new_name = f'{cat}{i}.{f_ext}'
        os.rename(j, new_name)


# to use this properly the sets must not be premixed. There is definitely a more elegant way of doing this.
"""
Currently, hardcoded for ease of use, just input the class/category str of the items and
it will throw them in a folder ext with the same name as the class/category.
set the integer to be a % of the total dataset volume. This can be done in code as well. 
To avoid conflicts or hard to see errors it 'seems' easier to just use a function to split data into separate folders.
"""

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
