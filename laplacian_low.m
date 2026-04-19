function filtered_signal = laplacian_low(signal, electrode_positions, channel_indices)
%LAPLACIAN_LOW Apply low Laplacian spatial filtering to multichannel EEG.
%
% Inputs:
%   signal               - EEG matrix [n_samples x n_channels]
%   electrode_positions  - Electrode coordinates [2 x n_channels]
%   channel_indices      - Indices of channels to filter
%
% Output:
%   filtered_signal      - Low-Laplacian-filtered EEG [n_samples x n_channels]

    num_samples = size(signal, 1);
    num_channels = length(channel_indices);
    filtered_signal = zeros(num_samples, num_channels);

    for channel_idx = 1:num_channels
        center_channel = channel_indices(channel_idx);
        center_position = electrode_positions(:, center_channel);

        distances = compute_distances(center_position, electrode_positions);
        [sorted_distances, sorted_indices] = sort(distances, 'ascend');

        % Exclude the center electrode itself; take the next 8 nearest
        candidate_indices = sorted_indices(2:9);
        candidate_positions = electrode_positions(:, candidate_indices);
        candidate_distances = sorted_distances(2:9);

        selected_local_indices = select_axis_neighbors(center_position, candidate_positions);
        neighbor_indices = candidate_indices(selected_local_indices);
        neighbor_distances = candidate_distances(selected_local_indices);

        weights = (1 ./ neighbor_distances) / sum(1 ./ neighbor_distances);
        weighted_neighbor_signal = signal(:, neighbor_indices) * weights(:);

        filtered_signal(:, channel_idx) = signal(:, center_channel) - weighted_neighbor_signal;
    end
end

function distances = compute_distances(center_position, all_positions)
    num_points = size(all_positions, 2);
    distances = zeros(1, num_points);

    for point_idx = 1:num_points
        distances(point_idx) = norm(center_position - all_positions(:, point_idx));
    end
end

function selected_indices = select_axis_neighbors(center_position, candidate_positions)
    x_distances = abs(candidate_positions(1, :) - center_position(1));
    y_distances = abs(candidate_positions(2, :) - center_position(2));

    [~, x_order] = sort(x_distances, 'ascend');
    [~, y_order] = sort(y_distances, 'ascend');

    selected_indices = unique([x_order(1:2), y_order(1:2)], 'stable');

    % Ensure exactly 4 neighbors
    if numel(selected_indices) < 4
        combined_order = unique([x_order, y_order], 'stable');
        for idx = combined_order
            if ~ismember(idx, selected_indices)
                selected_indices(end + 1) = idx; %#ok<AGROW>
            end
            if numel(selected_indices) == 4
                break;
            end
        end
    else
        selected_indices = selected_indices(1:4);
    end
end