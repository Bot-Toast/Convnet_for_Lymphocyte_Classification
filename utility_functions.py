import glob
import os
import random
import shutil
import numpy as np
import pandas as pds

# don't call these within main, create a separate project!

"""
Iterates through a chosen file directory renaming all the files to user specification.
rationale: To easily distribute large datasets into different collections, using the rand_valid_selector function.
I will probably forget how to use this in a week.
"""

"""
def file_renamer(cat, f_ext, file_path):
    os.chdir('E:\\Dissertation\\Dissertation_code\\find_the_bad_guys\\Datasets\\Train\\Normal lymphomcytes')
    print(os.getcwd())

    for i, j in enumerate(os.listdir()):
        new_name = f'{cat}{i}.{f_ext}'
        os.rename(j, new_name)
"""

# to use this properly the sets must not be premixed. There is definitely a more elegant way of doing this.
"""
Currently, hardcoded for ease of use, just input the class/category str of the items and
it will throw them in a folder ext with the same name as the class/category.
set the integer to be a % of the total dataset volume. This can be done in code as well. 
To avoid conflicts or hard to see errors it 'seems' easier to just use a function to split data into separate folders.
"""
"""

def rand_valid_selector(in_path, cat, out_path):
    dir_str = input("enter a directory to move images from: ")
    os.chdir('E:\\Dissertation\\Dissertation_code\\find_the_bad_guys\\Datasets')
    for i in random.sample(glob.glob(f'{in_path}\\{cat}*.jpg'), 44):
        shutil.move(i, f'{out_path}\\{cat}')


in_path = 'E:\\Dissertation\\Dissertation_code\\find_the_bad_guys\\Datasets\\train\\big\\'
out_path = 'E:\\Dissertation\\Dissertation_code\\find_the_bad_guys\\Datasets\\Validation\\'

klass = dir_str

print(klass)
print(in_path)
print(out_path)

rand_valid_selector(in_path, klass, out_path)
"""

"""
REMOVE RUN_INT FROM FUNCTIONS WHEN NOT LOOP TESTING
"""
# It's a SIN!
def prediction_result_appender(prediction_results, test_set):
    train_labels = ["non-stressed", "stressed"]
    df = pds.DataFrame()
    df.insert(0, 'file', str)
    df.insert(1, 'prediction', str)
    df.insert(2, 'percentage', str)
    np1: list[str] = []
    np2: list[str] = []

    for i in range(len(prediction_results)):
        score = prediction_results[i]
        score2: str = "{}".format(train_labels[np.argmax(score)])
        print(f"THIS IS SCORE2: {score2}")
        # cont = test_set.filepaths[i]
        score3: str = ("{:.2f}".format(100 * np.max(score)))
        print(f"THIS IS SCORE3: {score3}")
        np1.append(score2)
        np2.append(score3)
        print('dataframe columns: ', df.columns)
        print("this image is called: {} is likely classified as: {} with a: {:.2f} percent confidence".
              format(test_set.filepaths[i], train_labels[np.argmax(score)], 100 * np.max(score)))

    print(np.round(prediction_results))
    df['file'] = test_set.filepaths
    df['prediction'] = np1
    df['percentage'] = np2
    classes = np.argmax(prediction_results, axis=1)
    print(classes)
    df.to_csv(f"Run_{run_int}_post_norm_test_data_CNN_Results" + ".csv")
