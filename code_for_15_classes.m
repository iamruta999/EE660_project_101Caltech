% This code performs classification on 15 classes
clear all;
clc;
% CREATE BAG OF WORDS USING THE TRAINING DATA
% Get the images for all the classes and store them in different folders
%EDIT THE FOLLOWING LINES
% myFolder_1 = 'ENTER YOUR ADDRESS/101_ObjectCategories/airplanes';
% myFolder_2 = 'ENTER YOUR ADDRESS/101_ObjectCategories/Faces';
% myFolder_3 = 'ENTER YOUR ADDRESS/101_ObjectCategories/Motorbikes';
% myFolder_4 = 'ENTER YOUR ADDRESS/101_ObjectCategories/watch';
% myFolder_5 = 'ENTER YOUR ADDRESS/101_ObjectCategories/Leopards';
% myFolder_6 = 'ENTER YOUR ADDRESS/101_ObjectCategories/ketch';
% myFolder_7 = 'ENTER YOUR ADDRESS/101_ObjectCategories/chandelier';
% myFolder_8 = 'ENTER YOUR ADDRESS/101_ObjectCategories/bonsai';
% myFolder_9 = 'ENTER YOUR ADDRESS/101_ObjectCategories/car_side';
% myFolder_10 = 'ENTER YOUR ADDRESS/101_ObjectCategories/brain';
% myFolder_11 = 'ENTER YOUR ADDRESS/101_ObjectCategories/buddha';
% myFolder_12 = 'ENTER YOUR ADDRESS/101_ObjectCategories/butterfly';
% myFolder_13 = 'ENTER YOUR ADDRESS/101_ObjectCategories/ewer';
% myFolder_14 = 'ENTER YOUR ADDRESS/101_ObjectCategories/grand_piano';
% myFolder_15 = 'ENTER YOUR ADDRESS/101_ObjectCategories/hawksbill';
numberOfClasses = 15;
% Get a list of all files in those folders with the desired file name pattern.
filePattern_1 = fullfile(myFolder_1, '*.jpg'); 
theFiles_1 = dir(filePattern_1);

filePattern_2 = fullfile(myFolder_2, '*.jpg'); 
theFiles_2 = dir(filePattern_2);

filePattern_3 = fullfile(myFolder_3, '*.jpg'); 
theFiles_3 = dir(filePattern_3);

filePattern_4 = fullfile(myFolder_4, '*.jpg'); 
theFiles_4 = dir(filePattern_4);

filePattern_5 = fullfile(myFolder_5, '*.jpg'); 
theFiles_5 = dir(filePattern_5);

filePattern_6 = fullfile(myFolder_6, '*.jpg'); 
theFiles_6 = dir(filePattern_6);

filePattern_7 = fullfile(myFolder_7, '*.jpg'); 
theFiles_7 = dir(filePattern_7);

filePattern_8 = fullfile(myFolder_8, '*.jpg'); 
theFiles_8 = dir(filePattern_8);

filePattern_9 = fullfile(myFolder_9, '*.jpg'); 
theFiles_9 = dir(filePattern_9);

filePattern_10 = fullfile(myFolder_10, '*.jpg'); 
theFiles_10 = dir(filePattern_10);

filePattern_11 = fullfile(myFolder_11, '*.jpg'); 
theFiles_11 = dir(filePattern_11);

filePattern_12 = fullfile(myFolder_12, '*.jpg'); 
theFiles_12 = dir(filePattern_12);

filePattern_13 = fullfile(myFolder_13, '*.jpg'); 
theFiles_13 = dir(filePattern_13);

filePattern_14 = fullfile(myFolder_14, '*.jpg'); 
theFiles_14 = dir(filePattern_14);

filePattern_15 = fullfile(myFolder_15, '*.jpg'); 
theFiles_15 = dir(filePattern_15);

% TRAINING PHASE
% Count the number of images in each class and then choose the number of
% training images so that the training data is balanced
numberOfTrainingImages = min([length(theFiles_1),length(theFiles_2),length(theFiles_3),length(theFiles_4),length(theFiles_5),...
    length(theFiles_6),length(theFiles_7),length(theFiles_8),length(theFiles_9),length(theFiles_10),length(theFiles_11),...
    length(theFiles_12),length(theFiles_13),length(theFiles_14),length(theFiles_15)])-20;
%Extract SIFT features
fullDescriptors = zeros(0,128); % initialize a matrix to store all the descriptors together
for k = 1 : numberOfTrainingImages
    %For class 1
    image = imread(fullfile(myFolder_1,theFiles_1(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage) ;
%UNCOMMENT THE FOLLOWING COMMENTS TO DISPLAY THE IMAGE ALONG WITH THE
%SIFT FEATURES AND THE SELECTED DESCRIPTORS
%     figure;
%     imshow(image)
%     perm = randperm(size(features,2)) ;
%     sel = perm(1:20) ;
%     h1 = vl_plotframe(features(:,:)) ;% Plot the features on the image
%     h2 = vl_plotframe(features(:,:)) ;
%     set(h1,'color','k','linewidth',3) ;
%     set(h2,'color','y','linewidth',2) ;
%     h3 = vl_plotsiftdescriptor(descriptors(:,sel),features(:,sel)) ;% Plot a few descriptors
%     set(h3,'color','g') ;
    fullDescriptors = [fullDescriptors;descriptors'];%Concatenate the descriptors together
    %For class 2
    image = imread(fullfile(myFolder_2,theFiles_2(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage) ;
    fullDescriptors = [fullDescriptors;descriptors'];%Concatenate the descriptors together
    %For class 3
    image = imread(fullfile(myFolder_3,theFiles_3(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage) ;
    fullDescriptors = [fullDescriptors;descriptors'];%Concatenate the descriptors together
    %For class 4
    image = imread(fullfile(myFolder_4,theFiles_4(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage) ;
    fullDescriptors = [fullDescriptors;descriptors'];%Concatenate the descriptors together
    %For class 5
    image = imread(fullfile(myFolder_5,theFiles_5(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage) ;
    fullDescriptors = [fullDescriptors;descriptors'];%Concatenate the descriptors together
    %For class 6
    image = imread(fullfile(myFolder_6,theFiles_6(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage) ;
    fullDescriptors = [fullDescriptors;descriptors'];%Concatenate the descriptors together
    %For class 7
    image = imread(fullfile(myFolder_7,theFiles_7(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage) ;
    fullDescriptors = [fullDescriptors;descriptors'];%Concatenate the descriptors together
    %For class 8
    image = imread(fullfile(myFolder_8,theFiles_8(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage) ;
    fullDescriptors = [fullDescriptors;descriptors'];%Concatenate the descriptors together
    %For class 9
    image = imread(fullfile(myFolder_9,theFiles_9(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage) ;
    fullDescriptors = [fullDescriptors;descriptors'];%Concatenate the descriptors together
    %For class 10
    image = imread(fullfile(myFolder_10,theFiles_10(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage) ;
    fullDescriptors = [fullDescriptors;descriptors'];%Concatenate the descriptors together
    %For class 11
    image = imread(fullfile(myFolder_11,theFiles_11(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage) ;
    fullDescriptors = [fullDescriptors;descriptors'];%Concatenate the descriptors together
    %For class 12
    image = imread(fullfile(myFolder_12,theFiles_12(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage) ;
    fullDescriptors = [fullDescriptors;descriptors'];%Concatenate the descriptors together
    %For class 13
    image = imread(fullfile(myFolder_13,theFiles_13(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage) ;
    fullDescriptors = [fullDescriptors;descriptors'];%Concatenate the descriptors together
    %For class 14
    image = imread(fullfile(myFolder_14,theFiles_14(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage) ;
    fullDescriptors = [fullDescriptors;descriptors'];%Concatenate the descriptors together
    %For class 15
    image = imread(fullfile(myFolder_15,theFiles_15(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage) ;
    fullDescriptors = [fullDescriptors;descriptors'];%Concatenate the descriptors together
end

%Perform k-means clustering on the descriptors to get k centroids
numberOfClusters = 2000; % Define k
%[centroids, clusterNumber] = vl_kmeans(double(fullDescriptors'), numberOfClusters);
% CONVERT IMAGES INTO HISTOGRAM FEATURES 
% Get the histograms of the training images
fullHistogram = zeros(0,numberOfClusters); % initialize a matrix to store all the descriptors together
% Get the final features for class 1
for k = 1 : numberOfTrainingImages
    image = imread(fullfile(myFolder_1,theFiles_1(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    fullHistogram = [fullHistogram;imageHistogram];
end
% Get the final features for class 2
for k = 1 : numberOfTrainingImages
    image = imread(fullfile(myFolder_2,theFiles_2(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    fullHistogram = [fullHistogram;imageHistogram];
end
% Get the final features for class 3
for k = 1 : numberOfTrainingImages
    image = imread(fullfile(myFolder_3,theFiles_3(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    fullHistogram = [fullHistogram;imageHistogram];
end
% Get the final features for class 4
for k = 1 : numberOfTrainingImages
    image = imread(fullfile(myFolder_4,theFiles_4(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    fullHistogram = [fullHistogram;imageHistogram];
end
% Get the final features for class 5
for k = 1 : numberOfTrainingImages
    image = imread(fullfile(myFolder_5,theFiles_5(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    fullHistogram = [fullHistogram;imageHistogram];
end
% Get the final features for class 6
for k = 1 : numberOfTrainingImages
    image = imread(fullfile(myFolder_6,theFiles_6(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    fullHistogram = [fullHistogram;imageHistogram];
end
% Get the final features for class 7
for k = 1 : numberOfTrainingImages
    image = imread(fullfile(myFolder_7,theFiles_7(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    fullHistogram = [fullHistogram;imageHistogram];
end
% Get the final features for class 8
for k = 1 : numberOfTrainingImages
    image = imread(fullfile(myFolder_8,theFiles_8(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    fullHistogram = [fullHistogram;imageHistogram];
end
% Get the final features for class 9
for k = 1 : numberOfTrainingImages
    image = imread(fullfile(myFolder_9,theFiles_9(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    fullHistogram = [fullHistogram;imageHistogram];
end
% Get the final features for class 10
for k = 1 : numberOfTrainingImages
    image = imread(fullfile(myFolder_10,theFiles_10(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    fullHistogram = [fullHistogram;imageHistogram];
end
% Get the final features for class 11
for k = 1 : numberOfTrainingImages
    image = imread(fullfile(myFolder_11,theFiles_11(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    fullHistogram = [fullHistogram;imageHistogram];
end
% Get the final features for class 12
for k = 1 : numberOfTrainingImages
    image = imread(fullfile(myFolder_12,theFiles_12(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    fullHistogram = [fullHistogram;imageHistogram];
end
% Get the final features for class 13
for k = 1 : numberOfTrainingImages
    image = imread(fullfile(myFolder_13,theFiles_13(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    fullHistogram = [fullHistogram;imageHistogram];
end
% Get the final features for class 14
for k = 1 : numberOfTrainingImages
    image = imread(fullfile(myFolder_14,theFiles_14(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    fullHistogram = [fullHistogram;imageHistogram];
end
% Get the final features for class 15
for k = 1 : numberOfTrainingImages
    image = imread(fullfile(myFolder_15,theFiles_15(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    fullHistogram = [fullHistogram;imageHistogram];
end
% Create training labels according to the categories
total = numberOfTrainingImages * numberOfClasses;
labels = zeros(total,1);
for i = 1:numberOfClasses
labels((i-1)*numberOfTrainingImages+1:i*numberOfTrainingImages,:) = i;
end
% Train and Predict labels for training data
[result_Training] = multisvm(fullHistogram,labels,fullHistogram);
disp('Accuracy for training dataset is')
disp((sum(result_Training == labels)/total)*100);

% VALIDATION PHASE
validationHistogram = zeros(0,numberOfClusters);
% Test images from class 1
for k = numberOfTrainingImages+1:numberOfTrainingImages+12
        image = imread(fullfile(myFolder_1,theFiles_1(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    validationHistogram = [validationHistogram;imageHistogram];
end
    % Test images from class 2
for k = numberOfTrainingImages+1:numberOfTrainingImages+12
    image = imread(fullfile(myFolder_2,theFiles_2(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    validationHistogram = [validationHistogram;imageHistogram];
end
    % Test images from class 3
for k = numberOfTrainingImages+1:numberOfTrainingImages+12
    image = imread(fullfile(myFolder_3,theFiles_3(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    validationHistogram = [validationHistogram;imageHistogram];
end
% Test images from class 4
for k = numberOfTrainingImages+1:numberOfTrainingImages+12
    image = imread(fullfile(myFolder_4,theFiles_4(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    validationHistogram = [validationHistogram;imageHistogram];
end
% Test images from class 5
for k = numberOfTrainingImages+1:numberOfTrainingImages+12
    image = imread(fullfile(myFolder_5,theFiles_5(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    validationHistogram = [validationHistogram;imageHistogram];
end
% Test images from class 6
for k = numberOfTrainingImages+1:numberOfTrainingImages+12
    image = imread(fullfile(myFolder_6,theFiles_6(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    validationHistogram = [validationHistogram;imageHistogram];
end
% Test images from class 7
for k = numberOfTrainingImages+1:numberOfTrainingImages+12
    image = imread(fullfile(myFolder_7,theFiles_7(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    validationHistogram = [validationHistogram;imageHistogram];
end
% Test images from class 8
for k = numberOfTrainingImages+1:numberOfTrainingImages+12
    image = imread(fullfile(myFolder_8,theFiles_8(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    validationHistogram = [validationHistogram;imageHistogram];
end
% Test images from class 9
for k = numberOfTrainingImages+1:numberOfTrainingImages+12
    image = imread(fullfile(myFolder_9,theFiles_9(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    validationHistogram = [validationHistogram;imageHistogram];
end
% Test images from class 10
for k = numberOfTrainingImages+1:numberOfTrainingImages+12
    image = imread(fullfile(myFolder_10,theFiles_10(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    validationHistogram = [validationHistogram;imageHistogram];
end
% Test images from class 11
for k = numberOfTrainingImages+1:numberOfTrainingImages+12
    image = imread(fullfile(myFolder_11,theFiles_11(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    validationHistogram = [validationHistogram;imageHistogram];
end
% Test images from class 12
for k = numberOfTrainingImages+1:numberOfTrainingImages+12
    image = imread(fullfile(myFolder_12,theFiles_12(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    validationHistogram = [validationHistogram;imageHistogram];
end
% Test images from class 13
for k = numberOfTrainingImages+1:numberOfTrainingImages+12
    image = imread(fullfile(myFolder_13,theFiles_13(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    validationHistogram = [validationHistogram;imageHistogram];
end
% Test images from class 14
for k = numberOfTrainingImages+1:numberOfTrainingImages+12
    image = imread(fullfile(myFolder_14,theFiles_14(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    validationHistogram = [validationHistogram;imageHistogram];
end
% Test images from class 15
for k = numberOfTrainingImages+1:numberOfTrainingImages+12
    image = imread(fullfile(myFolder_15,theFiles_15(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    validationHistogram = [validationHistogram;imageHistogram];
end

% Create labels for validation data
numberOfValidationImages = 12;
totalValidationImages = numberOfValidationImages * numberOfClasses;
validationLabels = zeros(totalValidationImages,1);
for i = 1:numberOfClasses
validationLabels((i-1)*numberOfValidationImages+1:i*numberOfValidationImages,:) = i;
end
% Validating SVM
% Predict labels for validation data
[result_Validation] = multisvm(fullHistogram,labels,validationHistogram);
disp('Accuracy for validation dataset is')
disp((sum(result_Validation == validationLabels)/totalValidationImages)*100);

% TESTING PHASE
% Get histograms for test images by using the bag of words approach
testHistogram = zeros(0,numberOfClusters);
% Test images from class 1
for k = numberOfTrainingImages+14:numberOfTrainingImages+20
        image = imread(fullfile(myFolder_1,theFiles_1(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    testHistogram = [testHistogram;imageHistogram];
end
    % Test images from class 2
for k = numberOfTrainingImages+14:numberOfTrainingImages+20
    image = imread(fullfile(myFolder_2,theFiles_2(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    testHistogram = [testHistogram;imageHistogram];
end
    % Test images from class 3
for k = numberOfTrainingImages+14:numberOfTrainingImages+20
    image = imread(fullfile(myFolder_3,theFiles_3(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    testHistogram = [testHistogram;imageHistogram];
end
% Test images from class 4
for k = numberOfTrainingImages+14:numberOfTrainingImages+20
    image = imread(fullfile(myFolder_4,theFiles_4(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    testHistogram = [testHistogram;imageHistogram];
end
% Test images from class 5
for k = numberOfTrainingImages+14:numberOfTrainingImages+20
    image = imread(fullfile(myFolder_5,theFiles_5(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    testHistogram = [testHistogram;imageHistogram];
end
% Test images from class 6
for k = numberOfTrainingImages+14:numberOfTrainingImages+20
    image = imread(fullfile(myFolder_6,theFiles_6(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    testHistogram = [testHistogram;imageHistogram];
end
% Test images from class 7
for k = numberOfTrainingImages+14:numberOfTrainingImages+20
    image = imread(fullfile(myFolder_7,theFiles_7(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    testHistogram = [testHistogram;imageHistogram];
end
% Test images from class 8
for k = numberOfTrainingImages+14:numberOfTrainingImages+20
    image = imread(fullfile(myFolder_8,theFiles_8(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    testHistogram = [testHistogram;imageHistogram];
end
% Test images from class 9
for k = numberOfTrainingImages+14:numberOfTrainingImages+20
    image = imread(fullfile(myFolder_9,theFiles_9(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    testHistogram = [testHistogram;imageHistogram];
end
% Test images from class 10
for k = numberOfTrainingImages+14:numberOfTrainingImages+20
    image = imread(fullfile(myFolder_10,theFiles_10(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    testHistogram = [testHistogram;imageHistogram];
end
% Test images from class 11
for k = numberOfTrainingImages+14:numberOfTrainingImages+20
    image = imread(fullfile(myFolder_11,theFiles_11(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    testHistogram = [testHistogram;imageHistogram];
end
% Test images from class 12
for k = numberOfTrainingImages+14:numberOfTrainingImages+20
    image = imread(fullfile(myFolder_12,theFiles_12(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    testHistogram = [testHistogram;imageHistogram];
end
% Test images from class 13
for k = numberOfTrainingImages+14:numberOfTrainingImages+20
    image = imread(fullfile(myFolder_13,theFiles_13(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    testHistogram = [testHistogram;imageHistogram];
end
% Test images from class 14
for k = numberOfTrainingImages+14:numberOfTrainingImages+20
    image = imread(fullfile(myFolder_14,theFiles_14(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    testHistogram = [testHistogram;imageHistogram];
end
% Test images from class 15
for k = numberOfTrainingImages+14:numberOfTrainingImages+20
    image = imread(fullfile(myFolder_15,theFiles_15(k).name));
    if(size(image,3) ==3 )
        grayImage = single(rgb2gray(image)) ; % convert image to gray scale if its RGB
    else grayImage = single(image);
    end
    [features,descriptors] = vl_sift(grayImage);
    imageClusterNumber = zeros(1,size(descriptors,2));
    for i = 1:size(descriptors,2)
    [~, imageClusterNumber(i)] = min(vl_alldist(double(descriptors(:,i)), centroids)) ;
    end
    for i=1:numberOfClusters
    imageHistogram(i) = histc(imageClusterNumber,i);
    end
    testHistogram = [testHistogram;imageHistogram];
end

% Create labels for test data
numberOfTestingImages = 7;
totalTestImages = numberOfTestingImages * numberOfClasses;
testingLabels = zeros(totalTestImages,1);
for i = 1:numberOfClasses
testingLabels((i-1)*numberOfTestingImages+1:i*numberOfTestingImages,:) = i;
end
%Testing SVM
% Predict labels for testing data
[result_Testing] = multisvm(fullHistogram,labels,testHistogram);
disp('Accuracy for testing dataset is')
disp((sum(result_Testing == testingLabels)/totalTestImages)*100);