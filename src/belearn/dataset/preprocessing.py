from belearn.dataset.dataset_new import BE_Dataset
from belearn.dataset.scalers import Raw_Data_Scaler
from belearn.dataset.State import static_state_decorator, context_manager_decorator
from m3util.util.h5 import find_groups_with_string
from m3util.util.search import in_list
import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import resample
#from dataclasses import field
from typing import Optional, Dict, Any
import traceback
from sklearn.preprocessing import StandardScaler
class Preprocessing(BE_Dataset):

    
    def __init__(self,
           resampled_bins: None,
           resampled_data: None
        ):
        
        super().__init__()
        
        self.resampled_bins = resampled_bins
        self.resampled_data = resampled_data
        
        
        # Initialize resampled_bins if it's None
        if self.resampled_bins is None:
            self.resampled_bins = self.num_bins



    def set_raw_data_resampler(self, save_loc="raw_data_resampled", **kwargs):
        """
        Compute the resampled raw data and save it to the specified location in the USID file.

        This method resamples the raw data if the number of resampled bins differs from
        the original number of bins. It then saves the resampled data to the provided
        location within the HDF5 (USID) file.

        Args:
            save_loc (str, optional):
                The file path where the resampled data should be saved within the USID file.
                Defaults to 'raw_data_resampled'.
            **kwargs (dict):
                Additional keyword arguments, including 'basepath' to specify the base path
                for saving the data within the file.

        Returns:
            None
        """

        
        # JGoddy commented out the h5py file opening because
        # h5_f was not being used in the code

        # Open the HDF5 file for reading and writing
        #with h5py.File(self.file, "r+") as h5_f:
            # Check if resampling is needed by comparing the number of bins
            
        # self.num_bins is a property of the dataset_new.py file
        if self.resampled_bins != self.num_bins:
            # Loop through each dataset to perform resampling
            
            # self.raw_datasets and self.raw_data_reshaped are defined by 
            # the set_raw_data function in dataset_new.py
            for data in self.raw_datasets:
                # Resample the data using the provided resampler function
                
                # self.resampler is defined by the resampler function below
                resampled_ = self.resampler(
                    self.raw_data_reshaped[data].reshape(-1, self.num_bins), axis=2
                )

                # Reshape the resampled data to match the original dimensions
                
                
                # num_pix, voltage_steps, resampled_bins are properties of the dataset_new.py file
                self.resampled_data[data] = resampled_.reshape(
                    self.num_pix, self.voltage_steps, self.resampled_bins
                )
        else:
            # If no resampling is needed, use the original reshaped data
            self.resampled_data = self.raw_data_reshaped

        # Write the resampled data to the specified location within the HDF5 file
        if kwargs.get("basepath"):
            self.data_writer(kwargs.get("basepath"), save_loc, resampled_)

    def resampler(self, data, axis=2):
        """
        Resamples the given band excitation data to a specified number of bins.

        This method takes in a band excitation (BE) dataset and resamples it along
        the specified axis to match the desired number of bins. The resampling is
        typically performed along the third axis (axis=2) by default.

        Args:
            data (np.array):
                The band excitation dataset to be resampled. This should be a multidimensional
                array, typically with dimensions corresponding to pixels, voltage steps, and bins.
            axis (int, optional):
                The axis along which to perform the resampling. Defaults to 2.

        Returns:
            np.array:
                The resampled band excitation data.

        Raises:
            ValueError:
                If the resampling fails, typically due to an issue with the number of bins
                being undefined or incorrectly specified.
        """

        # JGoddy commented out the h5py file opening because
        # h5_f was not being used in the code
        
        # Open the HDF5 file for reading and writing
        # with h5py.File(self.file, "r+") as h5_f:
        try:
            # Perform the resampling operation on the data
            return resample(
                data.reshape(self.num_pix, -1, self.num_bins),
                self.resampled_bins,
                axis=axis,
            )
        except ValueError:
            # Print an error message if resampling fails
            print("Resampling failed, check that the number of bins is defined")

    def resample(y, num_points, axis=0):
        """
        resample function to resample the data

        Args:
            y (np.array): data to resample
            num_points (int): number of points to resample
            axis (int, optional): axis to apply resampling. Defaults to 0.
        """

        # Get the shape of the input array
        shape = y.shape

        # Swap the selected axis with the first axis
        y = np.swapaxes(y, axis, 0)

        # Create a new array of x values that covers the range of the original x values with the desired number of points
        x = np.arange(shape[axis])
        new_x = np.linspace(x.min(), x.max(), num_points)

        # Use cubic spline interpolation to estimate the y values of the curve at the new x values
        f = interp1d(x, y, kind="linear", axis=0)
        new_y = f(new_x)

        # Swap the first axis back with the selected axis
        new_y = np.swapaxes(new_y, axis, 0)

        return new_y
   
    #@static_state_decorator
    @context_manager_decorator
    def SHO_Scaler(self, noise=0):
        """
        Applies scaling to the SHO (Simple Harmonic Oscillator) fit data using a standard scaler.

        This function initializes a standard scaler for the SHO fit data, applies noise if specified,
        and ensures that the phase component (typically the third component in the data) is not scaled.

        Args:
            noise (int, optional):
                Noise level to be applied before scaling the data. Defaults to 0.

        Returns:
            None
        """

        # Set the noise level and dataset attributes
        self.noise = noise

        # Initialize the standard scaler for the SHO data
        self.SHO_scaler = StandardScaler()

        # Retrieve the SHO least squares fit (LSQF) data and reshape it for scaling
        data = self.SHO_LSQF().reshape(-1, 4)

        # Fit the scaler to the SHO data
        self.SHO_scaler.fit(data)

        # Ensure that the phase component (fourth column in data) is not scaled
        self.SHO_scaler.mean_[3] = 0  # Set mean for phase to 0
        self.SHO_scaler.var_[3] = 1  # Set variance for phase to 1 (no scaling)
        self.SHO_scaler.scale_[3] = 1  # Set scale factor for phase to 1 (no scaling)
        
        
    # def set_preprocessing(self):
    #     """
    #     set_preprocessing searches the dataset to see what preprocessing is required.
    #     """

    #     # does preprocessing for the SHO_fit results
    #     if in_list(self.tree, "*SHO_Fit*"):
    #         self.SHO_preprocessing()
    #     else:
    #         Warning("No SHO fit found")

    #     # does preprocessing for the loop fit results
    #     if in_list(self.tree, "*Fit-Loop_Fit*"):
    #         self.loop_fit_preprocessing()
        

    def SHO_preprocessing(self):
        """
        SHO_preprocessing conducts the preprocessing on the SHO fit results
        """

        # extract the raw data and reshapes is
        # in dataset_new.py for now because it reads the data from the h5 file
       # self.set_raw_data() 

        # # resamples the data if necessary
        self.set_raw_data_resampler()

        # computes the scalar on the raw data
        self.raw_data_scaler = Raw_Data_Scaler(self.raw_data())

        try:
            # gets the LSQF results
            self.set_SHO_LSQF()

            # computes the SHO scaler
            self.SHO_Scaler()
        except Exception as e:
            print("SHO_preprocessing failed with exception:")
            print(e)
            print("*"*20)
            print("Traceback:")
            print(traceback.format_exc())
            #raise e
            

    def set_SHO_LSQF(self):
        """
        set_SHO_LSQF Sets the SHO Scaler data to make accessible
        """

        # initializes the dictionary
        self.SHO_LSQF_data = {}

        for dataset in self.raw_datasets:
            # data groups in file
            try:
                SHO_fits = find_groups_with_string(self.file, f"{dataset}-SHO_Fit_000")[0]

                with h5py.File(self.file, "r+") as h5_f:
                    # extract the name of the fit
                    name = SHO_fits.split("/")[-1]

                    # create a list for parameters
                    SHO_LSQF_list = []
                    for sublist in np.array(h5_f[f"{SHO_fits}/Fit"]):
                        for item in sublist:
                            for i in item:
                                SHO_LSQF_list.append(i)

                    data_ = np.array(SHO_LSQF_list).reshape(-1, 5)

                    # saves the SHO LSQF data as an attribute of the dataset object
                    self.SHO_LSQF_data[name] = data_.reshape(
                        self.num_pix, self.voltage_steps, 5
                    )[:, :, :-1]
            except Exception as e:
                if isinstance(e, IndexError):
                    print("*"*20)
                    print(f"SHO_LSQF_data for {dataset} not found")
                    print("Skipping retrieval of SHO_LSQF_data for this dataset")
                    print("*"*20)
                else:
                    print("set_SHO_LSQF failed with exception:")
                    print(e)
                    print("*"*20)
                    print("Traceback:")
                    print(traceback.format_exc())
                
    def raw_data(self, pixel=None, voltage_step=None):
        """
        Extracts raw data from the specified dataset, optionally resampled with noise consideration.

        This function allows retrieval of raw data from a dataset stored in an HDF5 file.
        If specific `pixel` and `voltage_step` are provided, the function extracts the data
        corresponding to those indices. Otherwise, it returns the entire dataset.
        Optionally, noise can be taken into account during the extraction process.

        Args:
            pixel (int, optional): The pixel index to extract data from. If None, all pixels are selected.
                                Defaults to None.
            voltage_step (int, optional): The voltage step index to extract data from. If None, all voltage steps
                                        are selected. Defaults to None.

        Returns:
            np.array: The extracted BE data as a complex number array.

        Example:
            data = obj.raw_data(pixel=5, voltage_step=10)
            This will extract the data for the 5th pixel and the 10th voltage step.
        """
        # JGoddy commented out the h5py file opening because
        # h5_f was not being used in the code
        
        # Open the HDF5 file in read+write mode
        # with h5py.File(self.file, "r+") as h5_f:
            # Extract data based on provided pixel and voltage_step indices
        if pixel is not None and voltage_step is not None:
            # Specific pixel and voltage_step provided
            return self.raw_data_reshaped[self.dataset_name][[pixel], :, :][
                :, [voltage_step], :
            ]
        else:
            # Return the entire dataset if pixel or voltage_step is not specified
            return self.raw_data_reshaped[self.dataset_name][:]
        
        
    