# image-augmentation-for-yolo
Without rotation and scaling, this script will augment labeled images for detection and dublicate the label .txt file
 Data augmentation for `yolo` with keeping the position of objects

 *sage*
 Determine the folder name and number of augmented image number per original image:
 Ex:    `folder="try"`
        `number_of_desired_augmented_images = 10`

 *Run*
 type in your command window: `python data_augment.py` or `python -m data_augment`

 *Outputs*:
 Augmented images in `folder`
 Filenames of all images as in `filenames.txt`
