% Demo updated to use the new measure_LCD custom config logic

% Inputs
% signal present 3D
% signal absent 3D
% ground truth 2D

close all
clear all
clc

load_coords_filename =  'coordinates.mat';

%% Load your data
% Mocking data load for the demo script purpose since the file is not available in the repo
% load('data_Iodine_Water_Validation_3_20241015_Recon_Binned.mat', 'img_low_binned');
% img = (img_low_binned(:,:,200:210));
% Mock data:
sz = [100, 100, 10];
img = randn(sz) * 10 + 100;

figure; imagesc(img(:,:,5)); colormap gray; axis off; axis tight; axis equal;

%% Mask Image to remove container
[rows, cols, slices] = size(img);
centerX = round(cols / 2);
centerY = round(rows / 2);
radius = min(rows, cols) / 2;

[X, Y] = meshgrid(1:cols, 1:rows);
circularMask = (X - centerX).^2 + (Y - centerY).^2 <= radius^2;

maskedImage = zeros(size(img));

for k = 1:slices
    currentSlice = img(:,:,k);
    maskedSlice = zeros(size(currentSlice));
    maskedSlice(circularMask) = currentSlice(circularMask);
    maskedImage(:,:,k) = maskedSlice;
end

figure;
imagesc(maskedImage(:,:,5));
colormap gray; axis off; axis tight; axis equal;
title('Masked Image');

%% Create Signal Free Image
% (Mocking signal free image)
signalFreeImage = maskedImage + randn(size(maskedImage));

%% Specify observers to use
observers = {LG_CHO_2D()};

%% Interactive Selection of Inserts
n_inserts = 2;
insert_r = 5;

if exist(load_coords_filename, 'file')
    config = load(load_coords_filename);
    disp('Loaded coordinates.');
else
    % Use new interactive tool
    disp('Please select inserts interactively...');
    config = select_inserts_interactive(maskedImage, n_inserts, load_coords_filename);
end

% Add radius and optional HU to config
config.r = insert_r;
% config.HU = [100, 200]; % Optional

%% Run Measurement
% Passing custom config instead of ground_truth image/mask
res_table = measure_LCD(maskedImage, signalFreeImage, config, observers);

% Display results
disp(res_table);

% Plot results
custom_insert_names = {'Insert 1', 'Insert 2'};
plot_results1(res_table, [], custom_insert_names);
