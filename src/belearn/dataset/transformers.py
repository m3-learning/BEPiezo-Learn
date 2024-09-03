import numpy as np
import torch


def to_complex(data, axis=None):
        """
        to_complex function that converts data to complex

        Args:
            data (any): data to convert
            axis (int, optional): axis which the data is structured. Defaults to None.

        Returns:
            np.array: complex array of the BE response
        """

        # converts to an array
        if type(data) == list:
            data = np.array(data)

        # if the data is already in complex form return
        if to_complex(data):
            return data

        # if axis is not provided take the last axis
        if axis is None:
            axis = data.ndim - 1

        return np.take(data, 0, axis=axis) + 1j * np.take(data, 1, axis=axis)
    

def is_complex(data):
    """
    is_complex function to check if data is complex. If not complex makes it a complex number

    Args:
        data (any): input data

    Returns:
        any: array or tensor as a complex number
    """

    data = data[0]

    if type(data) == torch.Tensor:
        complex_ = data.is_complex()

    if type(data) == np.ndarray:
        complex_ = np.iscomplex(data)
        complex_ = complex_.any()

    return complex_