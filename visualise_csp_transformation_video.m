function visualise_csp_transformation_video( ...
    train_class_1, train_class_2, ...
    test_class_1, test_class_2, ...
    csp_filters)

%VISUALISE_CSP_TRANSFORMATION_VIDEO
% Create an animated visualization of CSP transformation and save it as:
%   - MP4 video
%   - GIF animation
%
% Inputs:
%   train_class_1 - training trials for class 1 [samples x channels x trials]
%   train_class_2 - training trials for class 2 [samples x channels x trials]
%   test_class_1  - test trials for class 1 [samples x channels x trials]
%   test_class_2  - test trials for class 2 [samples x channels x trials]
%   csp_filters   - CSP projection matrix [channels x num_filters]

    results_folder = 'results';

    if ~exist(results_folder, 'dir')
        mkdir(results_folder);
    end

    mp4_path = fullfile(results_folder, 'csp_animation.mp4');
    gif_path = fullfile(results_folder, 'csp_animation.gif');

    writer_obj = VideoWriter(mp4_path, 'MPEG-4');
    writer_obj.FrameRate = 10;
    open(writer_obj);

    is_first_gif_frame = true;

    figure('Name', 'CSP Transformation Animation');

    %% Training trials
    num_train_trials = min(size(train_class_1, 3), size(train_class_2, 3));
    num_features = size(csp_filters, 2);

    feature_train_1 = zeros(num_features, num_train_trials);
    feature_train_2 = zeros(num_features, num_train_trials);

    for trial_idx = 1:num_train_trials
        x1 = train_class_1(:, :, trial_idx)';   % [channels x samples]
        x2 = train_class_2(:, :, trial_idx)';

        % Before CSP
        subplot(2,2,1);
        plot(x1(53,:), x1(55,:), 'r.');
        hold on;
        plot(x2(53,:), x2(55,:), 'b.');
        title('EEG Before CSP');
        xlabel('Channel 53');
        ylabel('Channel 55');

        % Apply CSP
        y1 = csp_filters' * x1;
        y2 = csp_filters' * x2;

        % After CSP
        subplot(2,2,2);
        plot(y1(1,:), y1(2,:), 'r.');
        hold on;
        plot(y2(1,:), y2(2,:), 'b.');
        title('EEG After CSP');
        xlabel('CSP Component 1');
        ylabel('CSP Component 2');

        % Variance feature space
        feature_train_1(:, trial_idx) = var(y1, 0, 2);
        feature_train_2(:, trial_idx) = var(y2, 0, 2);

        subplot(2,2,3);
        plot(feature_train_1(1,:), feature_train_1(2,:), 'rs', ...
            'LineWidth', 2, 'MarkerSize', 8);
        hold on;
        plot(feature_train_2(1,:), feature_train_2(2,:), 'bo', ...
            'LineWidth', 2, 'MarkerSize', 8);
        title('Training Feature Space');
        xlabel('Feature 1');
        ylabel('Feature 2');

        drawnow;

        frame = getframe(gcf);
        writeVideo(writer_obj, frame);

        img = frame2im(frame);
        [imind, cm] = rgb2ind(img, 256);

        if is_first_gif_frame
            imwrite(imind, cm, gif_path, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
            is_first_gif_frame = false;
        else
            imwrite(imind, cm, gif_path, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
        end
    end

    %% Test trials
    num_test_trials = min(size(test_class_1, 3), size(test_class_2, 3));

    feature_test_1 = zeros(num_features, num_test_trials);
    feature_test_2 = zeros(num_features, num_test_trials);

    for trial_idx = 1:num_test_trials
        x1 = test_class_1(:, :, trial_idx)';   % [channels x samples]
        x2 = test_class_2(:, :, trial_idx)';

        % Before CSP
        subplot(2,2,1);
        plot(x1(53,:), x1(55,:), 'r.');
        hold on;
        plot(x2(53,:), x2(55,:), 'b.');
        title('EEG Before CSP');
        xlabel('Channel 53');
        ylabel('Channel 55');

        % Apply CSP
        y1 = csp_filters' * x1;
        y2 = csp_filters' * x2;

        % After CSP
        subplot(2,2,2);
        plot(y1(1,:), y1(2,:), 'r.');
        hold on;
        plot(y2(1,:), y2(2,:), 'b.');
        title('EEG After CSP');
        xlabel('CSP Component 1');
        ylabel('CSP Component 2');

        % Variance feature space
        feature_test_1(:, trial_idx) = var(y1, 0, 2);
        feature_test_2(:, trial_idx) = var(y2, 0, 2);

        subplot(2,2,4);
        plot(feature_test_1(1,:), feature_test_1(2,:), 'rs', ...
            'LineWidth', 2, 'MarkerSize', 8);
        hold on;
        plot(feature_test_2(1,:), feature_test_2(2,:), 'bo', ...
            'LineWidth', 2, 'MarkerSize', 8);
        title('Test Feature Space');
        xlabel('Feature 1');
        ylabel('Feature 2');

        drawnow;

        frame = getframe(gcf);
        writeVideo(writer_obj, frame);

        img = frame2im(frame);
        [imind, cm] = rgb2ind(img, 256);

        if is_first_gif_frame
            imwrite(imind, cm, gif_path, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
            is_first_gif_frame = false;
        else
            imwrite(imind, cm, gif_path, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
        end
    end

    close(writer_obj);
end