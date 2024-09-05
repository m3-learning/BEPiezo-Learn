import numpy as np
from autophyslearn.spectroscopic.nn import Multiscale1DFitter
import torch
from torch import nn

def MSE(true, prediction):
    """
    Computes the Mean Squared Error (MSE) between the true and predicted values.

    Args:
        true (numpy.ndarray): Ground truth values. It should be a NumPy array.
        prediction (numpy.ndarray): Predicted values. It should have the same shape as 'true'.

    Returns:
        numpy.ndarray or float: The MSE for each batch if there are multiple samples, or 
                                a scalar MSE value if there is only one sample.

    Notes:
        The function flattens all dimensions except the batch dimension for both 'true' and 'prediction' 
        arrays to compute the MSE per batch.
    """
    # Reshape the true and prediction arrays to 2D, preserving the batch size (first dimension)
    # This operation flattens all other dimensions (channels, timesteps, etc.)
    mse = np.mean((true.reshape(true.shape[0], -1) - prediction.reshape(true.shape[0], -1))**2, axis=1)

    # If there's only one batch (single sample), return MSE as a scalar
    if mse.shape[0] == 1:
        return mse.item()

    # Return the MSE for each batch
    return mse


def mse_rankings(true, prediction, curves=False):
    """
    Calculates the mean squared error (MSE) for the given predictions relative to the true values, 
    ranks them based on the error, and optionally returns the true and predicted values sorted by the ranked error.

    Args:
        true (array-like): Ground truth values. 
                          It should be convertible to a NumPy array, typically structured as [batch, channels, timesteps].
        prediction (array-like): Predicted values. 
                                 It should have the same structure as the 'true' values.
        curves (bool, optional): If True, return the sorted true and predicted values 
                                 along with the ranked errors. Default is False.

    Returns:
        index (numpy.ndarray): Indices of the predictions ranked by MSE in ascending order.
        errors (numpy.ndarray): MSE values corresponding to each prediction, sorted by rank.
        true (numpy.ndarray, optional): If 'curves' is True, returns the true values sorted by ranked MSE.
        prediction (numpy.ndarray, optional): If 'curves' is True, returns the predicted values sorted by ranked MSE.

    """
    def type_conversion(data):
        """
        Converts input data to a NumPy array and rearranges the axes so that
        the batch axis is moved to the end.

        Args:
            data (array-like): Input data to be converted.

        Returns:
            numpy.ndarray: Data with the batch axis moved to the last position.
        """
        data = np.array(data)
        data = np.rollaxis(data, 0, data.ndim - 1)  # Move batch axis to the end
        return data

    # Convert the input true and prediction arrays
    true = type_conversion(true)
    prediction = type_conversion(prediction)

    # Compute mean squared errors between the true and predicted values
    errors = MSE(prediction, true)

    # Rank predictions based on the MSE values in ascending order
    index = np.argsort(errors)

    # If curves is True, return ranked true and predicted values
    if curves:
        # true will be in the form [ranked error, channel, timestep]
        return index, errors[index], true[index], prediction[index]

    # Otherwise, return the indices and ranked errors only
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



def print_mse(model_obj, model_predictor, data, labels):
    """
    Prints the Mean Squared Error (MSE) of the model's predictions for each dataset provided.

    Args:
        model_obj: The object containing the dataset and any necessary methods for data extraction.
        model_predictor: The object or model responsible for making predictions on the input data.
        data (tuple): A tuple of datasets used to calculate the MSE. Each dataset can either be 
                      a PyTorch tensor or a dictionary containing data for prediction.
        labels (list): A list of strings corresponding to the names of the datasets, used for labeling the output.

    This function computes the MSE for each dataset in `data`, either by calling the `predict` method 
    of the `model_predictor` on tensor data or by extracting raw data from the `model_obj` for dictionary data.
    The MSE is computed for each dataset and printed with the corresponding label.
    """

    # Loop through each dataset and its corresponding label
    for data, label in zip(data, labels):

        # If the data is a PyTorch tensor
        if isinstance(data, torch.Tensor):
            # Compute predictions using the model's predict method
            pred_data, scaled_param, parm = model_predictor.predict(data)

        # If the data is a dictionary, use raw data extraction methods from model_obj
        elif isinstance(data, dict):
            # Extract raw data from LSQF SHO fits in the dataset
            pred_data, _ = model_obj.dataset.get_raw_data_from_LSQF_SHO(data)
            # Get true data in NN format from the dataset
            data, _ = model_obj.dataset.NN_data()
            # Convert predictions to a PyTorch tensor
            pred_data = torch.from_numpy(pred_data)

        # Uncomment if necessary: Conversion to hysteresis tensor format (currently disabled)
        # data = model_obj.dataset.hysteresis_tensor(data)
        # pred_data = model_obj.dataset.hysteresis_tensor(pred_data)

        # Compute the MSE between the true data and predicted data using PyTorch's MSELoss
        out = nn.MSELoss()(data, pred_data)

        # Print the MSE result with the dataset label
        print(f"{label} Mean Squared Error: {out:0.7f}")
