import numpy as np


def MSE(true, prediction):

        # calculates the mse
        mse = np.mean((true.reshape(
            true.shape[0], -1) - prediction.reshape(true.shape[0], -1))**2, axis=1)

        # converts to a scalar if there is only one value
        if mse.shape[0] == 1:
            return mse.item()

        return mse

def mse_rankings(true, prediction, curves=False):

        def type_conversion(data):

            data = np.array(data)
            data = np.rollaxis(data, 0, data.ndim-1)

            return data

        true = type_conversion(true)
        prediction = type_conversion(prediction)

        errors = MSE(prediction, true)

        index = np.argsort(errors)

        if curves:
            # true will be in the form [ranked error, channel, timestep]
            return index, errors[index], true[index], prediction[index]

        return index, errors[index]

def get_rankings(raw_data, pred, n=1, curves=True):
    """
    A simple function to get the best, median, and worst reconstructions based on MSE (mean squared error).
    
    This function ranks the predictions (`pred`) compared to the true values (`raw_data`) based on their MSE.
    It returns the indices of the best, median, and worst reconstructions, along with their corresponding MSE values
    and optionally the reconstruction curves.

    Args:
        raw_data (np.array): Array of the true values (ground truth).
        pred (np.array): Array of the predictions (reconstructed values).
        n (int, optional): Number of best, median, and worst reconstructions to return. Defaults to 1.
        curves (bool, optional): Whether to return the reconstruction curves for the selected indices. Defaults to True.

    Returns:
        ind (np.array): Indices of the best, median, and worst reconstructions.
        mse (np.array): MSE values corresponding to the best, median, and worst reconstructions.
        d1 (np.array): First set of reconstruction data for the selected indices (if `curves` is True).
        d2 (np.array): Second set of reconstruction data for the selected indices (if `curves` is True).
    """
    
    # Compute the rankings based on MSE and get the corresponding indices and MSE values.
    # If curves=True, it will also return the associated reconstruction curves (d1, d2).
    index, mse, d1, d2 = mse_rankings(raw_data, pred, curves=curves)
    
    # Calculate the index for the middle reconstruction (median).
    middle_index = len(index) // 2
    
    # Determine the range of indices to select for the median values.
    start_index = middle_index - n // 2
    end_index = start_index + n

    # Combine the best (first n), median (middle n), and worst (last n) indices.
    ind = np.hstack((index[:n], index[start_index:end_index], index[-n:])).flatten().astype(int)
    
    # Combine the corresponding MSE values for the best, median, and worst reconstructions.
    mse = np.hstack((mse[:n], mse[start_index:end_index], mse[-n:]))
    
    # Combine the reconstruction curves (d1, d2) for the best, median, and worst reconstructions.
    # Use squeeze to remove unnecessary dimensions from the resulting arrays.
    d1 = np.stack((d1[:n], d1[start_index:end_index], d1[-n:])).squeeze()
    d2 = np.stack((d2[:n], d2[start_index:end_index], d2[-n:])).squeeze()

    # Return the indices, MSE values, and optionally the reconstruction curves (d1, d2).
    return ind, mse, d1, d2
