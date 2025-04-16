from sklearn.model_selection import train_test_split
from belearn.dataset.State import State
from belearn.util.wrappers import context_manager_decorator
import torch
import numpy as np
from dataclasses import dataclass
#from belearn.viz.viz_new import Viz
#from autophyslearn.spectroscopic.nn import Model


@dataclass
class BE_model_utils(State):
##### Machine Learning Functions #####

    def __init__(self):
        super().__init__()
        #self.model = model
       # self.set_model_utils(self)
        


    #@static_state_decorator
    @context_manager_decorator
    def NN_data(self, resampled=None, noise = None,scaled=True):
        """
        Utility function that retrieves and prepares the data for neural network training.

        Args:
            resampled (bool, optional): If True, use the resampled data; otherwise, use original data. Defaults to None.
            scaled (bool, optional): If True, use scaled data; otherwise, use unscaled data. Defaults to True.
            noise (int, optional): If provided, use the specified noise level; otherwise, use the default noise level. Defaults to None.

        Returns:
            torch.tensor: A tuple containing:
                - x_data: Neural network input data.
                - y_data: Scaled SHO LSQF fit parameters (reshaped and converted to a tensor).
        """

        # Print the current state of the data extraction process
        self.extraction_state

        # If resampled is specified, ensure the correct dataset is used
        if resampled is not None:
            self.resampled = resampled

        self.noise = noise
        # Ensure the data is scaled if required, as scaling is often necessary for neural network training
        self.scaled = scaled

        # Retrieve the raw spectral data
        data = self.raw_spectra(noise=self.noise, scaled=self.scaled)

        # Convert the raw data into a format suitable for neural network input
        x_data = self.to_nn(data)

        # Retrieve the SHO fit results, which are scaled LSQF parameters
        y_data = self.SHO_fit_results().reshape(-1, 4)

        # Convert the LSQF results into a tensor for use in training and evaluation
        y_data = torch.tensor(y_data, dtype=torch.float32)

        # Return the neural network input data and corresponding fit parameters
        return x_data, y_data
    
    
    def test_train_split_(
        self, test_size=0.2, random_state=42, resampled=None, noise = None, scaled=True, shuffle=True
    ):
        """
        Utility function that performs the train-test split on the neural network data.

        Args:
            test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int, optional): Seed used by the random number generator to ensure reproducibility. Defaults to 42.
            resampled (bool, optional): If True, use resampled data; otherwise, use original data. Defaults to None.
            scaled (bool, optional): If True, use scaled data; otherwise, use unscaled data. Defaults to True.
            shuffle (bool, optional): If True, shuffle the data before splitting. Defaults to True.

        Returns:
            torch.tensor: X_train, X_test, y_train, y_test
                - X_train: Training data features.
                - X_test: Testing data features.
                - y_train: Training data labels.
                - y_test: Testing data labels.
        """

        # Retrieve the neural network data based on resampling and scaling options
        x_data, y_data = self.NN_data(resampled, noise, scaled)

        # Perform the train-test split using the specified test size, random state, and shuffle options
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x_data,
            y_data,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )

        self.extraction_state

        # Return the split datasets
        return self.X_train, self.X_test, self.y_train, self.y_test



    def to_nn(self, data):
        """
        Converts band excitation data into a form suitable for training a neural network.

        This utility function takes in band excitation data, typically in the form of real and
        imaginary components, and processes it into a tensor format that can be used as input
        for neural networks. If the data is already a PyTorch tensor, it returns the data as is.

        Args:
            data (tuple or torch.Tensor): Band excitation data, typically as a tuple of
                                        (real, imag) or directly as a PyTorch tensor.

        Returns:
            torch.Tensor: A tensor with the real and imaginary components stacked along
                        a new dimension, ready for neural network training.
        """

        # If data is already a PyTorch tensor, return it as is.
        if type(data) == torch.Tensor:
            return data

        # Determine the number of bins based on whether the data has been resampled or not.
        if self.resampled: 
            bins = self.resampled_bins
        else:
            bins = self.num_bins

        # Unpack the real and imaginary parts of the data.
        real, imag = data

        # Reshape the real and imaginary components to have dimensions of samples x timesteps.
        real = real.reshape(-1, bins)
        imag = imag.reshape(-1, bins)

        # Stack the real and imaginary components along a new axis.
        # The result is a 3D array where the third dimension contains the real and imaginary parts.
        x_data = np.stack((real, imag), axis=2)

        # Convert the stacked array to a PyTorch tensor with the appropriate data type.
        x_data = torch.tensor(x_data, dtype=torch.float32)

        return x_data