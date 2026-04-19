function plot_laplacian_neighbors(electrode_positions, center_channel, neighbor_indices, plot_title)
%PLOT_LAPLACIAN_NEIGHBORS Visualize a center electrode and its selected neighbors.
%
% Inputs:
%   electrode_positions - Electrode coordinates [2 x n_channels]
%   center_channel      - Index of center electrode
%   neighbor_indices    - Indices of selected neighbor electrodes
%   plot_title          - Figure title string

    center_position = electrode_positions(:, center_channel);
    neighbor_positions = electrode_positions(:, neighbor_indices);

    figure;
    plot(electrode_positions(1, :), electrode_positions(2, :), 'ob', ...
        'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
    hold on;

    plot(center_position(1), center_position(2), 'sr', ...
        'LineWidth', 2, 'MarkerSize', 12, 'MarkerFaceColor', 'r');

    plot(neighbor_positions(1, :), neighbor_positions(2, :), 'sg', ...
        'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'g');

    title(plot_title);
    grid on;
    hold off;
end