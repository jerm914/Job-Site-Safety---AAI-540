import argparse
#import os
#import requests
#import tempfile
import numpy as np
import pandas as pd
#from sklearn.pipeline import Pipeline
import subprocess

# define function to copy files for respective data splits to corresponding s3 destinations
# provide 'split_name' as either 'train', 'val', 'test', or 'batch'
# provide 'split_list' as either 'train_list', 'val_list', 'test_list', or 'batch_list'
def split_dataset(split_name, split_list):
    
    # counter for printing copy progress
    num_files_copied = 0

    # iterate through each sample in split
    for index, sample in data[split_list].iterrows():

        # source/destination variables for individual sample
        cp_image_source = f"{s3_images_source}{sample['img_filename']}"
        cp_image_dest = f"{s3_split_dest}{split_name}/images/"
        cp_label_source = f"{s3_labels_source}{sample['label_filename']}"
        cp_label_dest = f"{s3_split_dest}{split_name}/labels/"

        # copy from source to destination
        #!aws s3 cp $cp_image_source $cp_image_dest --exclude "*" --include "*.jpg" --only-show-errors
        #!aws s3 cp $cp_label_source $cp_label_dest --exclude "*" --include "*.txt" --only-show-errors
        """
        subprocess.run(f"aws s3 cp {cp_image_source} {cp_image_dest} --exclude '*' --include '*.jpg' --only-show-errors", shell=True)
        subprocess.run(f"aws s3 cp {cp_label_source} {cp_label_dest} --exclude '*' --include '*.txt' --only-show-errors", shell=True)
        
        # increment counter
        num_files_copied += 1

        # print status after every 500 files copied
        if num_files_copied % 10 == 0:
            print(f"{num_files_copied} files copied.")
        """


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--s3-images-source', type=str)
    parser.add_argument('--s3-labels-source', type=str)
    parser.add_argument('--s3-split-dest', type=str)
    
    # obtain arguments
    args, _ = parser.parse_known_args()
    
    base_dir = "/opt/ml/processing"
    
    data = pd.read_csv(
        f"{base_dir}/input/catalog_query.csv"
    )
    
    # data split in four sets - training, validation, test, and batch inference
    rand_split = np.random.rand(len(data))
    train_list = rand_split < 0.4
    val_list = (rand_split >= 0.4) & (rand_split < 0.5)
    test_list = (rand_split >= 0.5) & (rand_split < 0.6)
    batch_list = rand_split >= 0.6 # "production" data

    # print data splits
    print('Data Splits:')
    print('------------')
    print(f"Train :   {sum(train_list)} samples")
    print(f"Val   :   {sum(val_list)} samples")
    print(f"Test  :   {sum(test_list)} samples")
    print(f"Batch :   {sum(batch_list)} samples")
    
    # define and print source s3 locations
    #s3_images_source = f"s3://{default_bucket}/{prefix_data}/images/"
    #s3_labels_source = f"s3://{default_bucket}/{prefix_data}/labels/"
    s3_images_source = args.s3_images_source
    s3_labels_source = args.s3_labels_source
    print('Images source directory location:', s3_images_source)
    print('Labels source directory location:', s3_labels_source, '\n')

    # define and print destination s3 location for data splits
    s3_split_dest = args.s3_split_dest
    print('Split destination directory location:', s3_split_dest)
    
    # perform data copies
    print('Beginning TRAIN data split copies.')
    split_dataset(split_name='train', split_list=train_list)
    print('Completed TRAIN data split copies.\n')

    print('Beginning VAL data split copies.')
    split_dataset(split_name='val', split_list=val_list)
    print('Completed VAL data split copies.\n')

    print('Beginning TEST data split copies.')
    split_dataset(split_name='test', split_list=test_list)
    print('Completed TEST data split copies.\n')

    print('Beginning BATCH data split copies.')
    split_dataset(split_name='batch', split_list=batch_list)
    print('Completed BATCH data split copies.')
