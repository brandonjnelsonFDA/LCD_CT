% Demo for using Custom Phantom with LCD_CT toolkit
% This demo shows how to define custom insert locations and run the LCD analysis.

clear; clc; close all;

% 1. Create Synthetic Data (similar to other demos)
% For this demo we create a simple phantom with known inserts
sz = [100, 100, 10];
background = zeros(sz) + 100; % Background HU
inserts = zeros(sz);

% Define inserts
n_inserts = 2;
centers_x = [30, 70];
centers_y = [50, 50];
radius = 10;
insert_hu = [110, 120];

% Create ground truth and signal present images
ground_truth = background;
for i = 1:n_inserts
    [X, Y] = meshgrid(1:sz(2), 1:sz(1));
    mask = ((X - centers_x(i)).^2 + (Y - centers_y(i)).^2) <= radius^2;
    % Apply to all slices for simplicity
    for z = 1:sz(3)
        slice_mask = inserts(:,:,z);
        slice_mask(mask) = insert_hu(i) - 100;
        inserts(:,:,z) = slice_mask;

        slice_gt = ground_truth(:,:,z);
        slice_gt(mask) = insert_hu(i);
        ground_truth(:,:,z) = slice_gt;
    end
end

signal_present = background + inserts + randn(sz) * 5; % Add noise
signal_absent = background + randn(sz) * 5;

% 2. Define Custom Configuration
% Instead of relying on automatic detection or hardcoded MITA parameters,
% we define the insert locations manually.

% Option A: Programmatic definition
custom_config.x = centers_x;
custom_config.y = centers_y;
custom_config.r = radius; % Scalar (same for all) or vector
custom_config.HU = insert_hu;

disp('Running LCD analysis with programmatic custom configuration...');
observers = {'LG_CHO_2D'};
results = measure_LCD(signal_present, signal_absent, custom_config, observers);

disp('Results (Programmatic):');
disp(results);

% Option B: Interactive Selection (Commented out for automated testing)
% disp('Select 2 inserts interactively...');
% interactive_coords = select_inserts_interactive(signal_present, 2, 'my_coords.mat');
% interactive_config = interactive_coords;
% interactive_config.r = 10; % Radius needs to be known/set
% interactive_config.HU = [0, 0]; % HU optional
% results_interactive = measure_LCD(signal_present, signal_absent, interactive_config, observers);
% disp(results_interactive);
