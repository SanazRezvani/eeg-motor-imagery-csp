clc;
clear;
close all;

%% EEG Motor Imagery Baseline Pipeline
% Baseline MATLAB pipeline for motor imagery EEG classification
% using BCI Competition III - Dataset IVa.

%% Configuration
config.dataset_path = 'data_set_IVa_al.mat';
config.frequency_band = 'mu';        % options: 'mu', 'mu_beta', 'mu_beta_gamma'
config.spatial_filter = 'CAR';  % options: 'CAR', 'Low Laplacian', 'High Laplacian'
config.filter_order = 3;
config.train_ratio = 0.70;
config.num_csp_pairs = 1;
config.trial_length_s = 3.5;
config.plot_figures = true;
config.visualise_csp = true;

%% Load dataset

% Official dataset fields:
%   cnt      : continuous EEG signal [time x channels]
%   mrk.pos  : cue positions in samples
%   mrk.y    : trial labels
%   nfo.fs   : sampling rate
%   nfo.clab : channel labels
%   nfo.xpos : x positions of electrodes
%   nfo.ypos : y positions of electrodes

load(config.dataset_path);

%% Convert dataset variables into readable pipeline variables
raw_eeg_signal = 0.1 * double(cnt);   % convert to microvolts
sampling_rate = nfo.fs;
channel_labels = nfo.clab;
channel_x_positions = nfo.xpos;
channel_y_positions = nfo.ypos;
electrode_positions = [channel_x_positions'; channel_y_positions'];

trial_start_samples = mrk.pos;
trial_labels = mrk.y;
trial_length_samples = round(config.trial_length_s * sampling_rate);

results_folder = 'results';

if ~exist(results_folder, 'dir')
    mkdir(results_folder);
end

%% Plot channel layout
if config.plot_figures
    figure;
    plot(channel_x_positions, channel_y_positions, 'ro', ...
        'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'k');
    grid on;
    text(channel_x_positions + 0.02, channel_y_positions, channel_labels, ...
        'FontSize', 15, 'Color', 'b');
    title('EEG Channel Locations');
end

saveas(gcf, fullfile(results_folder, 'channel_layout.png'));

%% Select band-pass range
switch config.frequency_band
    case 'mu'
        low_cutoff_hz = 8;
        high_cutoff_hz = 13;

    case 'mu_beta'
        low_cutoff_hz = 8;
        high_cutoff_hz = 30;

    case 'mu_beta_gamma'
        low_cutoff_hz = 8;
        high_cutoff_hz = 49.9;   % limited by Nyquist when fs = 100 Hz

    otherwise
        error('Unknown frequency band option: %s', config.frequency_band);
end

%% Band-pass filtering
normalized_cutoff = [low_cutoff_hz high_cutoff_hz] / (sampling_rate / 2);
[b, a] = butter(config.filter_order, normalized_cutoff, 'bandpass');
bandpassed_signal = filtfilt(b, a, raw_eeg_signal);

%% Plot example time-domain signal
if config.plot_figures
    figure;
    subplot(2,1,1);
    plot(raw_eeg_signal(:, 1), 'r', 'LineWidth', 0.5);
    title('Raw EEG Signal - First Channel');

    subplot(2,1,2);
    plot(bandpassed_signal(:, 1), 'b', 'LineWidth', 0.5);
    title('Band-Pass Filtered EEG Signal - First Channel');
end

saveas(gcf, fullfile(results_folder, 'time_domain_signals.png'));

%% Plot frequency-domain signal
if config.plot_figures
    num_samples = size(raw_eeg_signal, 1);
    frequency_axis = linspace(0, sampling_rate / 2, round(num_samples / 2));

    raw_fft = abs(fft(raw_eeg_signal(:, 1)));
    raw_fft = raw_fft(1:round(num_samples / 2));

    filtered_fft = abs(fft(bandpassed_signal(:, 1)));
    filtered_fft = filtered_fft(1:round(num_samples / 2));

    figure;
    subplot(2,1,1);
    stem(frequency_axis, raw_fft, 'Marker', 'none', 'LineWidth', 1);
    title('Raw EEG Spectrum - First Channel');

    subplot(2,1,2);
    stem(frequency_axis, filtered_fft, 'Marker', 'none', 'LineWidth', 1);
    title('Filtered EEG Spectrum - First Channel');
end

saveas(gcf, fullfile(results_folder, 'frequency_spectrum.png'));

%% Apply spatial filtering
spatially_filtered_signal = apply_spatial_filter( ...
    bandpassed_signal, ...
    config.spatial_filter, ...
    electrode_positions);

if config.plot_figures
    figure;
    plot(spatially_filtered_signal(:, 1), 'r');
    title('Spatially Filtered EEG Signal - First Channel');
end

saveas(gcf, fullfile(results_folder, 'spatial_filtered_signal.png'));

%% Extract trials by class
% Dataset IVa competition labels:
%   1 = right
%   2 = foot
class_1_label = 1;
class_2_label = 2;
class_1_name = 'Right';
class_2_name = 'Foot';

class_1_trials = [];
class_2_trials = [];
class_1_count = 0;
class_2_count = 0;

for trial_idx = 1:length(trial_labels)
    current_label = trial_labels(trial_idx);

    if isnan(current_label)
        continue;
    end

    start_sample = trial_start_samples(trial_idx);
    end_sample = start_sample + trial_length_samples - 1;
    sample_indices = start_sample:end_sample;

    trial_signal = spatially_filtered_signal(sample_indices, :);

    if current_label == class_1_label
        class_1_count = class_1_count + 1;
        class_1_trials(:, :, class_1_count) = trial_signal;
    elseif current_label == class_2_label
        class_2_count = class_2_count + 1;
        class_2_trials(:, :, class_2_count) = trial_signal;
    end
end

%% Split each class into training and test sets
num_class_1_trials = size(class_1_trials, 3);
num_class_2_trials = size(class_2_trials, 3);

num_train_class_1 = floor(config.train_ratio * num_class_1_trials);
num_train_class_2 = floor(config.train_ratio * num_class_2_trials);

train_class_1 = class_1_trials(:, :, 1:num_train_class_1);
train_class_2 = class_2_trials(:, :, 1:num_train_class_2);

test_class_1 = class_1_trials(:, :, num_train_class_1 + 1:end);
test_class_2 = class_2_trials(:, :, num_train_class_2 + 1:end);

%% Compute CSP filters
csp_filters = compute_csp_filters(train_class_1, train_class_2, config.num_csp_pairs);

% Visualise CSP transformation (optional)
% Make "config.visualise_csp = true" at the top to see this visualisation
if config.plot_figures && config.visualise_csp
        visualise_csp_transformation_video( ...
        train_class_1, train_class_2, ...
        test_class_1, test_class_2, ...
        csp_filters)
end

%% Extract variance-based CSP features
train_features_class_1 = extract_csp_variance_features(train_class_1, csp_filters);
train_features_class_2 = extract_csp_variance_features(train_class_2, csp_filters);
test_features_class_1 = extract_csp_variance_features(test_class_1, csp_filters);
test_features_class_2 = extract_csp_variance_features(test_class_2, csp_filters);

%% Build feature matrices and labels
train_features = [train_features_class_1, train_features_class_2];
train_labels = [ ...
    ones(1, size(train_features_class_1, 2)), ...
    2 * ones(1, size(train_features_class_2, 2)) ...
];

test_features = [test_features_class_1, test_features_class_2];
test_labels = [ ...
    ones(1, size(test_features_class_1, 2)), ...
    2 * ones(1, size(test_features_class_2, 2)) ...
];

%% Train classifiers and evaluate performance
results = struct();

% Support Vector Machine (SVM)
results.svm.model = fitcsvm(train_features', train_labels');
results.svm.predictions = predict(results.svm.model, test_features');
results.svm.accuracy = mean(results.svm.predictions' == test_labels) * 100;

figure;
confusionchart(test_labels, results.svm.predictions);
title(sprintf('Confusion Matrix - SVM (%s vs %s)', class_1_name, class_2_name));

fprintf('Accuracy (SVM): %.2f%%\n', results.svm.accuracy);

saveas(gcf, fullfile(results_folder, 'confusion_svm.png'));

% K-Nearest Neighbors (KNN)
num_neighbors = 5;
results.knn.model = fitcknn(train_features', train_labels', ...
    'NumNeighbors', num_neighbors);
results.knn.predictions = predict(results.knn.model, test_features');
results.knn.accuracy = mean(results.knn.predictions' == test_labels) * 100;

figure;
confusionchart(test_labels, results.knn.predictions);
title(sprintf('Confusion Matrix - KNN (k = %d)', num_neighbors));

fprintf('Accuracy (KNN): %.2f%%\n', results.knn.accuracy);

saveas(gcf, fullfile(results_folder, 'confusion_knn.png'));

% Linear Discriminant Analysis (LDA)
results.lda.model = fitcdiscr(train_features', train_labels', ...
    'DiscrimType', 'pseudoLinear');
results.lda.predictions = predict(results.lda.model, test_features');
results.lda.accuracy = mean(results.lda.predictions' == test_labels) * 100;

figure;
confusionchart(test_labels, results.lda.predictions);
title('Confusion Matrix - LDA');

fprintf('Accuracy (LDA): %.2f%%\n', results.lda.accuracy);

saveas(gcf, fullfile(results_folder, 'confusion_lda.png'));

%% Compare classifier accuracies
classifier_names = categorical({'SVM', 'KNN', 'LDA'});
classifier_names = reordercats(classifier_names, {'SVM', 'KNN', 'LDA'});

accuracy_values = [
    results.svm.accuracy, ...
    results.knn.accuracy, ...
    results.lda.accuracy
];

figure;
bar(classifier_names, accuracy_values);
ylabel('Accuracy (%)');
title('Classifier Performance Comparison');
grid on;

saveas(gcf, fullfile(results_folder, 'accuracy_comparison.png'));

%% Local helper function
function features = extract_csp_variance_features(trials, csp_filters)
%EXTRACT_CSP_VARIANCE_FEATURES Extract variance-based features after CSP projection.
%
% Inputs:
%   trials       - EEG trials [samples x channels x trials]
%   csp_filters  - CSP projection matrix [channels x num_filters]
%
% Output:
%   features     - Feature matrix [num_filters x num_trials]

    num_trials = size(trials, 3);
    num_filters = size(csp_filters, 2);
    features = zeros(num_filters, num_trials);

    for trial_idx = 1:num_trials
        trial_data = trials(:, :, trial_idx)';   % [channels x samples]
        projected_trial = csp_filters' * trial_data;
        features(:, trial_idx) = var(projected_trial, 0, 2);
    end
end