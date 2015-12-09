$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$ Classification of a subset of 101 Caltech dataset $
$ Amruta Kulkarni				    $
$ Email: arkulkar@usc.edu			    $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

This repository contains the source code to implement image classification on a subset of images from the 10 Caltech dataset.The following steps will guide you to implement the codes contained in this repository.
1. Download the 101 Caltech dataset from the following link: http://www.vision.caltech.edu/Image_Datasets/Caltech101/
2. Save this folder in MATLAB (version 2014b or higher is preferred) and add its path
3. Download the vlfeat library from the following link:
http://www.vlfeat.org/
4. Save this folder in MATLAB and rename it as vlfeat and add this folder and its subfolders to the MATLAB path
5. Open MATLAB and type the following command:
run('vlfeat/toolbox/vl_setup');
6. Now copy the multisvm.m file and the two files named code_for_10_classes.m and code_for_15_classes.m into the MATLAB Documents folder and add them to the path.
7. Open code_for_10_classes.m. Edit lines 7 to 16 and put your address of MATLAB Documents folder. For example:
myFolder_1 = 'C:/Users/amruta/Documents/MATLAB/101_ObjectCategories/airplanes';
8. Now the code is ready to be used.Execute this file and the accuracy for training, validation and test dataset will get displayed in 50-60 minutes.
9. Open code_for_15_classes.m. Edit lines 7 to 21 and put your address of MATLAB Documents folder.
10. Now the code is ready to be used.Execute this file and the accuracy for training, validation and test dataset will get displayed in 2.5-3 hours.