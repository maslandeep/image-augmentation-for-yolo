#=================================================================
# 7/12/21
# Data augmentation for yolo with keeping the position of objects
# Melih Aslan
# maslanky@gmail.com

# Usage:
# Determine the folder name and number of augmented image number per original image:
# Ex:    folder="try"
#        number_of_desired_augmented_images = 10

# Run:
# type in your command window: "python data_augment_cleaned.py" or "python -m data_augment_cleaned"

# Outputs:
# Augmented images in "folder"
# Filenames of all images as in filenames.txt
#=================================================================

import cv2
import os
import numpy as np
from skimage.util import random_noise

from numpy import expand_dims
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

# this function will write the image paths in txt file
def write_image_filenames(folder):

    with open("filenames.txt", "w") as f1:
   
        for filename in os.listdir(folder):

            if filename[-1] == "g":
           
                f1.write(folder+"/"+filename)
                f1.write("\n")
               
               
        f1.close()
           
# this function will change the brightness of images between 0.7 and 1.5
def change_brightnes(img):

    # convert to numpy array
    data = img_to_array(img)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(brightness_range=[0.7,1.5])
    # prepare iterator
    it = datagen.flow(samples, batch_size=1)
    # generate batch of images
    batch = it.next()
    # convert to unsigned integers for viewing
    image = batch[0].astype('uint8')
   
    return image

# this function will dublicate label file for the augmented (not scaled or translated) images
def copy_text_to_another(folder_in, folder_out):
    with open(folder_in) as f:
        with open(folder_out, "w") as f1:
            for line in f:
                #print(line)
                #if "ROW" in line:
                f1.write(line)
            f1.close()

# this function will run the 
def augment_images(folder, N_iter):
   
    for filename in os.listdir(folder):

        if filename[-1] == "g":
       
            img = cv2.imread(os.path.join(folder,filename))
            keep_img = img
            h = img.shape[0]
            w = img.shape[1]
            z = img.shape[2]        
            dh = round(h/(0.7*N_iter)) # this number will be used for occlusion

            
            # lets do first 30% of augmentation with noise and brightness
            index_noise = round(0.3*N_iter)
            for iter_aug in range(index_noise):

 
                img = cv2.imread(os.path.join(folder,filename))
                noise_img = random_noise(img, mode='s&p',amount=0.1)
                noise_img = np.array(255*noise_img, dtype = 'uint8')
                #noise_img = cv2.cvtColor(noise_img, cv2.COLOR_BGR2GRAY)
                noise_img = change_brightnes(noise_img)
           
                # saving augmented image
                no = iter_aug
                path_out_img = folder + "/" + filename[:-4] + "-" + str(no) + ".jpg"
                cv2.imwrite(path_out_img, noise_img)

                # dublicating the labeled file for saved augmented image                
                path_in = folder + "/" + filename[:-4] + ".txt"
                path_out = folder + "/" + filename[:-4] + "-" + str(no) + ".txt"
                copy_text_to_another(path_in, path_out)            
            

            # lets do rest of augmentation with occlusion and brightness
            for iter_aug in range(index_noise, N_iter):

                img = cv2.imread(os.path.join(folder,filename))
                occ_constant = iter_aug - index_noise - 1
                occ_img = img
                occ_img[round(occ_constant*dh):(occ_constant+1)*dh][:] = 0
                occ_img = change_brightnes(occ_img)        

                # saving augmented image
                no = iter_aug
                path = folder + "/" + filename[:-4] + "-" + str(no) + ".jpg"
                cv2.imwrite(path, occ_img)
            
                # dublicating the labeled file for saved augmented image                            
                path_in = folder + "/" + filename[:-4] + ".txt"
                path_out = folder + "/" + filename[:-4] + "-" + str(no) + ".txt"
                copy_text_to_another(path_in, path_out)        
           
            
            
if __name__ == "__main__":
    folder="try"
    number_of_desired_augmented_images = 10
    augment_images(folder, number_of_desired_augmented_images) 
    write_image_filenames(folder)