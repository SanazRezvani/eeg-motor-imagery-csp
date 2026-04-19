function filtered_signal = apply_spatial_filter(signal, filter_type, electrode_positions)
%APPLY_SPATIAL_FILTER Apply a selected spatial filter to multichannel EEG.
%
% Inputs:
%   signal               - EEG matrix [n_samples x n_channels]
%   filter_type          - 'CAR', 'Low Laplacian', or 'High Laplacian'
%   electrode_positions  - Electrode coordinates [2 x n_channels]
%
% Output:
%   filtered_signal      - Spatially filtered EEG signal

    channel_indices = 1:size(signal, 2);

    switch filter_type
        case 'CAR'
            filtered_signal = car_filter(signal);

        case 'Low Laplacian'
            filtered_signal = laplacian_low(signal, electrode_positions, channel_indices);

        case 'High Laplacian'
            filtered_signal = laplacian_high(signal, electrode_positions, channel_indices);

        otherwise
            error('Unknown spatial filter type: %s', filter_type);
    end
end