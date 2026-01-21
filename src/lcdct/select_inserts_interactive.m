function coords = select_inserts_interactive(image_volume, n_inserts, output_filename)
% SELECT_INSERTS_INTERACTIVE Interactively select insert centers from an image.
%
%   coords = SELECT_INSERTS_INTERACTIVE(image_volume, n_inserts)
%   coords = SELECT_INSERTS_INTERACTIVE(image_volume, n_inserts, output_filename)
%
%   Inputs:
%       image_volume: 3D or 2D image array. If 3D, the center slice is used for display.
%       n_inserts: Number of inserts to select.
%       output_filename: (Optional) Path to save the coordinates as a .mat file.
%
%   Outputs:
%       coords: Struct containing 'x' and 'y' vectors of selected coordinates.

    if nargin < 2
        error('Usage: select_inserts_interactive(image_volume, n_inserts, [output_filename])');
    end

    % Handle 3D volume by taking center slice
    if ndims(image_volume) == 3
        slice_idx = round(size(image_volume, 3) / 2);
        img_display = image_volume(:, :, slice_idx);
    elseif ndims(image_volume) == 2
        img_display = image_volume;
    else
        error('Input image must be 2D or 3D');
    end

    % Display image
    f = figure;
    imagesc(img_display);
    colormap gray;
    axis off; axis tight; axis equal;
    title(sprintf('Select %d insert locations (Left click to select)', n_inserts));

    disp(sprintf('Please select %d insert locations on the figure window...', n_inserts));

    % Get inputs
    [x, y] = ginput(n_inserts);

    close(f);

    coords.x = x;
    coords.y = y;

    % Save if filename provided
    if nargin >= 3 && ~isempty(output_filename)
        save(output_filename, '-struct', 'coords');
        disp(['Coordinates saved to ', output_filename]);
    end
end
