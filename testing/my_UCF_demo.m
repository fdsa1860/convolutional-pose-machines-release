

%% Demo code of "Convolutional Pose Machines", 
% Shih-En Wei, Varun Ramakrishna, Takeo Kanade, Yaser Sheikh
% In CVPR 2016
% Please contact Shih-En Wei at shihenw@cmu.edu for any problems or questions
%%
close all;
addpath('src'); 
addpath('util');
addpath('util/ojwoodford-export_fig-5735e6d/');
param = config();

fprintf('Description of selected model: %s \n', param.model(param.modelID).description);

%% Edit this part
% put your own test image here
%test_image = 'sample_image/singer.jpg';
%test_image = 'sample_image/shihen.png';
%test_image = 'sample_image/roger.png';
%test_image = 'sample_image/nadal.png';
%test_image = 'sample_image/LSP_test/im1640.jpg';
%test_image = 'sample_image/CMU_panoptic/00000998_01_01.png';
%test_image = 'sample_image/CMU_panoptic/00004780_01_01.png';
%test_image = 'sample_image/FLIC_test/princess-diaries-2-00152201.jpg';
%interestPart = 'Lwri'; % to look across stages. check available names in config.m

dataPath = '../dataset/UCF_sports/diving/001';

files = dir(fullfile(dataPath, '*.jpg'));

%% core: apply model on the image, to get heat maps and prediction coordinates

detection = cell(1, length(files));
%for i = 1:2
for i = 1:length(files)
imName = fullfile(dataPath, files(i).name);
im = imread(imName);
rectangle = [1 1 size(im, 2) size(im, 1)];
[heatMaps, prediction] = my_applyModel(imName, param, rectangle);
detection{i} = prediction;
end

save diving001.mat detection
%% visualize, or extract variable heatMaps & prediction for your use
%visualize(test_image, heatMaps, prediction, param, rectangle, interestPart);
