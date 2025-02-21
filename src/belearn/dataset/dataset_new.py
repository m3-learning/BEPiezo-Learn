import os
import numpy as np
import time
import sidpy
from BGlib import be as belib

from m3util.util.h5 import (
    #find_groups_with_string,
    print_tree,
    get_tree,
    find_measurement,
    make_group,
)
import pyUSID as usid

import h5py

from dataclasses import dataclass
from typing import Optional
from belearn.util.wrappers import static_state_decorator


#functions in BE_Dataset class: 
# get_tree
# print_be_tree
# get_original_data
# num_pix
# num_bins
# voltage_steps
# spectroscopic_length
# set_raw_data

@dataclass
class BE_Dataset:
    file: str =  '/home/jca92/Rapid-Fitting-BEPFM-NN/notebooks/Data/data_raw.h5' #TODO: make required
    noise: int = 0
    """
    A class to represent a h5 file.

    Attributes:
        file (str): The path to the h5 file.
    """
    def __post_init__(self):
        
        #self.noise = self.noise_state
        self.tree = self.get_tree() 
        self.get_dataset(self.noise) 
    
        
    def get_dataset(self, noise):
        """Property that returns the current dataset based on the noise state."""

        if noise == 0:
            return "Raw_Data"
        else:
            return f"Noisy_Data_{noise}"
    
    @property
    def get_tree(self):
        """
        get_tree reads the tree from the H5 file

        Returns:
            list: list of the tree from the H5 file
        """

        with h5py.File(self.file, "r+") as h5_f:
            return get_tree(h5_f)
        
    @property
    def print_be_tree(self):
        """Utility file to print the Tree of a BE Dataset

        Code adapted from pyUSID

        Args:
            path (str): path to the h5 file
        """

        with h5py.File(self.file, "r+") as h5_f:
            # Inspects the h5 file
            usid.hdf_utils.print_tree(h5_f)

            # prints the structure and content of the file
            print(
                "Datasets and datagroups within the file:\n------------------------------------"
            )
            print_tree(h5_f.file)

            print("\nThe main dataset:\n------------------------------------")
            print(h5_f)
            print("\nThe ancillary datasets:\n------------------------------------")
            print(h5_f.file["/Measurement_000/Channel_000/Position_Indices"])
            print(h5_f.file["/Measurement_000/Channel_000/Position_Values"])
            print(h5_f.file["/Measurement_000/Channel_000/Spectroscopic_Indices"])
            print(h5_f.file["/Measurement_000/Channel_000/Spectroscopic_Values"])

            print(
                "\nMetadata or attributes in a datagroup\n------------------------------------"
            )

            for key in h5_f.file["/Measurement_000"].attrs:
                print("{} : {}".format(key, h5_f.file["/Measurement_000"].attrs[key]))
                
                
    @property
    def Raw_SHO_Data(self):
        """
        Retrieves the original raw Band Excitation (BE) data as a complex number array.

        This property accesses the raw data from an HDF5 file. Depending on the dataset
        specified, it either retrieves the data directly from the 'Raw_Data' dataset or
        searches for a dataset that matches a noise-specific naming convention.

        Returns:
            np.array: The BE data as a complex number array.

        Example:
            data = obj.get_original_data
            This will retrieve the raw BE data from the HDF5 file.
        """

        # Open the HDF5 file in read+write mode
        with h5py.File(self.file, "r+") as h5_f:
            # Check if the dataset is 'Raw_Data'
            if self.dataset == "Raw_Data":
                # Directly return the 'Raw_Data' from the HDF5 file
                return h5_f["Measurement_000"]["Channel_000"]["Raw_Data"][:]
            else:
                # If not 'Raw_Data', find the dataset that matches the noise-specific name
                name = find_measurement(
                    self.file, f"original_data_{self.noise}STD", group=self.basegroup
                )
                # Return the matched dataset
                return h5_f["Measurement_000"]["Channel_000"][name][:]        
            
    @property
    def num_pix(self):
        """Number of pixels in the data"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["num_pix"] 
    
    @property
    def num_bins(self):
        """Number of frequency bins in the data"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["num_bins"]
        
    @property
    def frequency_bin(self):
        """Frequency bin vector in Hz"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"]["Channel_000"]["Bin_Frequencies"][:]
        
    @property
    def voltage_steps(self):
        """Number of DC voltage steps"""
        with h5py.File(self.file, "r+") as h5_f:
            try:
                return h5_f["Measurement_000"].attrs["num_udvs_steps"]
            except:
                # computes the number of voltage steps for datasets that do not contain the attribute
                return (
                    h5_f["Measurement_000"].attrs["VS_steps_per_full_cycle"]
                    * h5_f["Measurement_000"].attrs["VS_number_of_cycles"]
                    * (
                        2
                        if h5_f["Measurement_000"].attrs["VS_measure_in_field_loops"]
                        == "in and out-of-field"
                        else 1
                    )
                )    
        
    @property
    def spectroscopic_length(self):
        """Gets the length of the spectroscopic vector"""
        return self.num_bins * self.voltage_steps    
    
    @property
    def get_pos_dims(self):
        """
        Retrieves the position dimensions of the main dataset from the HDF5 file.

        This property accesses the specified HDF5 file and extracts the position dimension
        information from the main dataset. It returns a list of `usid.Dimension` objects
        that describe each positional dimension in terms of its descriptor, label, and size.

        Returns:
            list of usid.Dimension: A list containing the position dimensions of the dataset.

        Example:
            pos_dims = obj.get_pos_dims
            for dim in pos_dims:
                print(f"Dimension Name: {dim.name}, Size: {dim.size}, Units: {dim.units}")
        """
        # Open the HDF5 file in read+write mode
        with h5py.File(self.file, "r+") as h5_f:
            # Find the main dataset named 'Raw_Data' within the HDF5 file
            h5_main = usid.hdf_utils.find_dataset(h5_f, "Raw_Data")[0]

            # Extract position dimension descriptors, labels, and sizes from the main dataset
            pos_dim_descriptors = h5_main.pos_dim_descriptors
            pos_dim_labels = h5_main.pos_dim_labels
            pos_dim_sizes = h5_main.pos_dim_sizes

            # Create the list of usid.Dimension objects
            pos_dim = [
                usid.Dimension(descriptor, label, size)
                for descriptor, label, size in zip(
                    pos_dim_descriptors, pos_dim_labels, pos_dim_sizes
                )
            ]

            # Return the list of position dimensions
            return pos_dim

    @property
    def get_spec_dims(self):
        """
        Retrieves the spectroscopic dimensions of the main dataset from the HDF5 file.

        This property accesses the specified HDF5 file and extracts the spectroscopic dimension
        information from the main dataset. It returns a list of `usid.Dimension` objects
        that describe each spectroscopic dimension in terms of its descriptor, label, and size.

        Returns:
            list of usid.Dimension: A list containing the spectroscopic dimensions of the dataset.

        Example:
            spec_dims = obj.get_spec_dims
            for dim in spec_dims:
                print(f"Dimension Name: {dim.name}, Size: {dim.size}, Units: {dim.units}")
        """

        # Open the HDF5 file in read+write mode
        with h5py.File(self.file, "r+") as h5_f:
            # Find the main dataset named 'Raw_Data' within the HDF5 file
            h5_main = usid.hdf_utils.find_dataset(h5_f, "Raw_Data")[0]

            # Extract spectroscopic dimension descriptors, labels, and sizes from the main dataset
            spec_dim_descriptors = h5_main.spec_dim_descriptors
            spec_dim_labels = h5_main.spec_dim_labels
            spec_dim_sizes = h5_main.spec_dim_sizes

            # Create the list of usid.Dimension objects
            spec_dim = [
                usid.Dimension(descriptor, label, size)
                for descriptor, label, size in zip(
                    spec_dim_descriptors, spec_dim_labels, spec_dim_sizes
                )
            ]

            return spec_dim

              
            
    def generate_noisy_data_records(
        self,
        noise_levels,
        basegroup="/Measurement_000/Channel_000",
        verbose=False,
        noise_STD=None,
    ):
        """
        Generates noisy data records and saves them to an HDF5 file.

        This function creates new datasets with added noise based on the provided noise levels
        and saves these noisy datasets to the specified group in the HDF5 file. The noise
        can be generated with a provided standard deviation or calculated from the original data.

        Args:
            noise_levels (list): A list of noise levels (multipliers) to apply to the dataset.
            basegroup (str, optional): The HDF5 group where the noisy datasets will be saved.
                                    Defaults to '/Measurement_000/Channel_000'.
            verbose (bool, optional): If True, the function will print additional information
                                    during execution. Defaults to False.
            noise_STD (float, optional): A manually provided standard deviation for the noise.
                                        If not provided, it will be calculated from the original data.
                                        Defaults to None.

        Example:
            obj.generate_noisy_data_records(noise_levels=[0.1, 0.2, 0.5], verbose=True)
            This will generate and save noisy datasets for the specified noise levels.
        """

        # Compute the noise standard deviation if it is not provided
        if noise_STD is None:
            noise_STD = np.std(self.Raw_SHO_Data)

        if verbose:
            print(f"The STD of the data is: {noise_STD}")

        # Open the HDF5 file in read+write mode
        with h5py.File(self.file, "r+") as h5_f:
            # Iterate through each noise level provided in the list
            for noise_level in noise_levels:
                if verbose:
                    print(f"Adding noise level {noise_level}")

                # Calculate the actual noise level to be applied
                noise_level_ = noise_STD * noise_level

                # Generate random noise for the real and imaginary parts
                noise_real = np.random.uniform(
                    -1 * noise_level_,
                    noise_level_,
                    (self.num_pix, self.spectroscopic_length),
                )
                noise_imag = np.random.uniform(
                    -1 * noise_level_,
                    noise_level_,
                    (self.num_pix, self.spectroscopic_length),
                )

                # Combine real and imaginary components to create complex noise
                noise = noise_real + noise_imag * 1.0j

                # Add the generated noise to the original data
                data = self.Raw_SHO_Data + noise

                # Find the original dataset in the HDF5 file
                h5_main = usid.hdf_utils.find_dataset(h5_f, "Raw_Data")[0]

                # Write the noisy data to the HDF5 file
                usid.hdf_utils.write_main_dataset(
                    h5_f[basegroup],  # Parent group where data is saved
                    data,  # Noisy data to be written
                    f"Noisy_Data_{noise_level}",  # Name for the noisy dataset
                    "Piezoresponse",  # Physical quantity being measured
                    "V",  # Units of the measurement
                    self.get_pos_dims,  # Position dimensions
                    self.get_spec_dims,  # Spectroscopic dimensions
                    h5_pos_inds=h5_main.h5_pos_inds,  # Position indices
                    h5_pos_vals=h5_main.h5_pos_vals,  # Position values
                    h5_spec_inds=h5_main.h5_spec_inds,  # Spectroscopic indices
                    h5_spec_vals=h5_main.h5_spec_vals,  # Spectroscopic values
                    compression="gzip",
                )  # Compression type for storage
                
    # this should maybe go in a separate 'preprocessing' class 
    def SHO_Fitter(
        self,
        force=False,
        max_cores=-1,
        max_mem=1024 * 8,
        dataset="Raw_Data",
        h5_sho_targ_grp=None,
        fit_group=False,
    ):
        """
        Computes the SHO (Simple Harmonic Oscillator) fit results for a given dataset.

        This function performs fitting of band excitation data using a SHO model. It
        leverages the BGlib library to handle the fitting process and saves the results
        to an HDF5 file. The function can be configured to use multiple cores and
        limit memory usage.

        Args:
            force (bool, optional):
                If True, forces the SHO results to be recomputed from scratch. Defaults to False.
            max_cores (int, optional):
                Number of processor cores to use for the fitting process. If -1, all available
                cores are used. Defaults to -1.
            max_mem (int, optional):
                Maximum amount of RAM (in MB) to use. Defaults to 1024*8 (8 GB).
            dataset (str, optional):
                Name of the dataset within the HDF5 file to be fitted. Defaults to "Raw_Data".
            h5_sho_targ_grp (h5py.Group, optional):
                The HDF5 group where the SHO fit results should be saved. If None, results
                are saved in the root group. Defaults to None.
            fit_group (bool, optional):
                If True, returns the SHO fitter object and fit results group. Defaults to False.

        Returns:
            belib.analysis.BESHOfitter:
                The SHO fitter object used to perform the fitting.
            h5py.Group, optional:
                The HDF5 group containing the SHO fit results. Returned only if `fit_group=True`.

        Raises:
            ValueError:
                If the fitting process encounters an error or if the necessary attributes
                cannot be found in the dataset.
        """

        with h5py.File(self.file, "r+") as h5_file:
            # Record the start time for the fitting process
            start_time_lsqf = time.time()

            # Split the directory path and the file name from the full file path
            (data_dir, filename) = os.path.split(self.file)

            if self.file.endswith(".h5"):
                # If the file is an HDF5 file, set the HDF5 path
                h5_path = self.file
            else:
                pass  # Handle non-HDF5 files if necessary

            # Split the path to get the folder and raw file name
            folder_path, h5_raw_file_name = os.path.split(h5_path)

            print("Working on:\n" + h5_path)

            # Get the main dataset to be fitted
            h5_main = usid.hdf_utils.find_dataset(h5_file, dataset)[0]

            # Extract useful parameters from the dataset
            pos_ind = h5_main.h5_pos_inds
            pos_dims = h5_main.pos_dim_sizes
            pos_labels = h5_main.pos_dim_labels
            print(pos_labels, pos_dims)

            # Get the measurement group containing the dataset
            h5_meas_grp = h5_main.parent.parent

            # Get all attributes of the measurement group
            parm_dict = sidpy.hdf_utils.get_attributes(h5_meas_grp)

            # Get the data type of the dataset
            expt_type = usid.hdf_utils.get_attr(h5_file, "data_type")

            # Check if the dataset is cKPFMData and set relevant parameters
            is_ckpfm = expt_type == "cKPFMData"
            if is_ckpfm:
                num_write_steps = parm_dict["VS_num_DC_write_steps"]
                num_read_steps = parm_dict["VS_num_read_steps"]
                num_fields = 2

            # Handle non-BELineData types
            if expt_type != "BELineData":
                vs_mode = usid.hdf_utils.get_attr(h5_meas_grp, "VS_mode")
                try:
                    field_mode = usid.hdf_utils.get_attr(
                        h5_meas_grp, "VS_measure_in_field_loops"
                    )
                except KeyError:
                    print("Field mode could not be found. Setting to default value.")
                    field_mode = "out-of-field"
                try:
                    vs_cycle_frac = usid.hdf_utils.get_attr(
                        h5_meas_grp, "VS_cycle_fraction"
                    )
                except KeyError:
                    print(
                        "VS cycle fraction could not be found. Setting to default value."
                    )
                    vs_cycle_frac = "full"

            # Set parameters for the SHO fitting process
            sho_fit_points = 5  # Number of data points to use when fitting
            sho_override = force  # Whether to force recompute if True

            # Determine the file path for saving the SHO fit results
            h5_sho_file_path = os.path.join(folder_path, h5_raw_file_name)
            print("\n\nSHO Fits will be written to:\n" + h5_sho_file_path + "\n\n")

            # Determine the file opening mode
            f_open_mode = "w" if not os.path.exists(h5_sho_file_path) else "r+"
            h5_sho_file = h5py.File(h5_sho_file_path, mode=f_open_mode)

            # Set the target group for saving SHO results
            if h5_sho_targ_grp is None:
                h5_sho_targ_grp = h5_sho_file
            else:
                h5_sho_targ_grp = make_group(h5_file, h5_sho_targ_grp)

            # Initialize the SHO fitter using the specified parameters
            sho_fitter = belib.analysis.BESHOfitter(
                h5_main, cores=max_cores, verbose=False, h5_target_group=h5_sho_targ_grp
            )

            # Set up the initial guess for the SHO fitting
            sho_fitter.set_up_guess(
                guess_func=belib.analysis.be_sho_fitter.SHOGuessFunc.complex_gaussian,
                num_points=sho_fit_points,
            )

            # Perform the initial guess fitting
            h5_sho_guess = sho_fitter.do_guess(override=sho_override)

            # Set up the actual fitting process
            sho_fitter.set_up_fit()

            # Perform the SHO fitting
            h5_sho_fit = sho_fitter.do_fit(override=sho_override)

            # Retrieve and print the fitting parameters
            parms_dict = sidpy.hdf_utils.get_attributes(h5_main.parent.parent)
            print(
                f"LSQF method took {time.time() - start_time_lsqf} seconds to compute parameters"
            )

            # Return the fitter and fit results if requested
            if fit_group:
                return sho_fitter, h5_sho_fit
            else:
                return sho_fitter


    #@static_state_decorator
    def set_raw_data(self):
        """
        set_raw_data Function that parses the datafile and extracts the raw data names
        """

        with h5py.File(self.file, "r+") as h5_f:
            # initializes the dictionary
            self.raw_data_reshaped = {}

            # list of datasets to be read
            datasets = []
            self.raw_datasets = []

            # Finds all the datasets
            datasets.extend(
                usid.hdf_utils.find_dataset(
                    h5_f["Measurement_000/Channel_000"], "Noisy"
                )
            )
            datasets.extend(
                usid.hdf_utils.find_dataset(
                    h5_f["Measurement_000/Channel_000"], "Raw_Data"
                )
            )

            # loops around all the datasets and stores them reshaped in a dictionary
            for dataset in datasets:
                self.raw_data_reshaped[dataset.name.split("/")[-1]] = dataset[
                    :
                ].reshape(self.num_pix, self.voltage_steps, self.num_bins)

                self.raw_datasets.extend([dataset.name.split("/")[-1]])

