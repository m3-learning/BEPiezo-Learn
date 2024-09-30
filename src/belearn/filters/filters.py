import numpy as np
from scipy.interpolate import interp1d

def clean_interpolate(arr, axis=0, method='cubic', verbose=False):
    """
    Function that removes bad data points by interpolating them.

    Args:
        arr (np.array): NumPy array to clean.
        axis (int, optional): Axis along which to interpolate. Defaults to 0.
        method (str, optional): Interpolation method ('linear', 'nearest', 'cubic'). Defaults to 'linear'.
        debug (bool, optional): If True, print debugging information. Defaults to False.

    Raises:
        ValueError: Raised if the selected axis is out of bounds.

    Returns:
        np.array: Cleaned array with bad data points interpolated.
    """
    # Check if the axis is valid
    if axis < 0 or axis >= arr.ndim:
        raise ValueError(f"Axis {axis} is out of bounds for the array.")

    # Move interpolation axis to the front
    arr_transposed = np.moveaxis(arr, axis, 0)
    
    # Reshape to 2D, keeping all dimensions except the selected axis flat
    original_shape = arr_transposed.shape
    arr_flat = arr_transposed.reshape(arr_transposed.shape[0], -1)

    # Process each "slice" along the flattened axis
    for i in range(arr_flat.shape[1]):
        slice_ = arr_flat[:, i]
        finite_mask = np.isfinite(slice_)
        finite_indices = np.where(finite_mask)[0]
        non_finite_indices = np.where(~finite_mask)[0]

        if len(finite_indices) < 2:
            # If there are fewer than 2 finite points, skip interpolation
            if debug:
                print(f"Skipping slice {i} due to insufficient finite data points.")
            continue

        if len(non_finite_indices) == 0:
            # No need to interpolate if there are no bad data points
            continue

        # Perform interpolation on non-finite values
        interp_func = interp1d(finite_indices, slice_[finite_indices], kind=method, bounds_error=False, fill_value="extrapolate")
        slice_[non_finite_indices] = interp_func(non_finite_indices)

        if debug:
            print(f"Interpolated {len(non_finite_indices)} bad data points in slice {i}.")

    # Reshape back to the original shape
    arr_cleaned = arr_flat.reshape(original_shape)

    # Move the axis back to its original position
    return np.moveaxis(arr_cleaned, 0, axis)
