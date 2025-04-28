import os
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

import time
import sidpy
from BGlib import be as belib

from m3util.util.h5 import (
    # find_groups_with_string,
    print_tree,
    get_tree,
    find_measurement,
    make_group,
)
import pyUSID as usid

import h5py
from pyUSID.io.hdf_utils import reshape_to_n_dims, get_auxiliary_datasets


from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path
from belearn.util.wrappers import static_state_decorator, context_manager_decorator
from belearn.filters.filters import clean_interpolate


from belearn.dataset.Datafed import BE_DataFed


@dataclass
class BE_Dataset(BE_DataFed):
    file: str = "./Data/data_raw.h5" 
    noise: int = 0
    resampled_bins: int = None
    resampled_data: dict = None
    datafed: Optional[Union[None, str, Path]] = None
    basegroup: str = "/Measurement_000/Channel_000"
    raw_data_path: str = "Raw_Data_SHO_Fit/Raw_Data-SHO_Fit_000"
    measurement_data_path: str = "Measurement_Data/Measurement_Data-000"
    measurement: str = "Measurement_000"
    SHO_fit_relative_base_path: str = "SHO_Fit_000"
    SHO_hysteresis_loop_fit_name: str = "Fit-Loop_Fit_000"
    SHO_hysteresis_loop_guess_name: str = "Guess-Loop_Fit_000"
    noise_std_: float = None
    """
    A class to represent a h5 file.

    Attributes:
        file (str): The path to the h5 file.
        resampled_bins (int): The number of bins to resample the data to.
        resampled_data (dict): The data to resample.
    """

    def __post_init__(self, datafed=None):
        # super().__init__(datafed)

        # TODO: why does this inherit from BE_DataFed?
        self.datafed = datafed
        self.get_dataset(self.noise)

        # self.resampled_bins = self.resampled_bins
        # self.resampled_data = self.resampled_data
        # # Initialize resampled_bins if it's None
        # if self.resampled_bins is None:
        #     self.resampled_bins = self.num_bins

    def get_dataset(self, noise):
        """Property that returns the current dataset based on the noise state."""

        if noise == 0:
            self.dataset_name = "Raw_Data"
        else:
            self.dataset_name = f"Noisy_Data_{noise}"

    @property
    def tree(self):
        """
        tree reads the tree from the H5 file

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
            print(h5_f.file[f"{self.basegroup}/Position_Indices"])
            print(h5_f.file[f"{self.basegroup}/Position_Values"])
            print(h5_f.file[f"{self.basegroup}/Spectroscopic_Indices"])
            print(h5_f.file[f"{self.basegroup}/Spectroscopic_Values"])

            print(
                "\nMetadata or attributes in a datagroup\n------------------------------------"
            )

            for key in h5_f.file[self.measurement].attrs:
                print("{} : {}".format(key, h5_f.file[self.measurement].attrs[key]))

    # This function was called get_original_data in the old code
    @property
    def raw_SHO_data(self):
        """
        Retrieves the original raw Band Excitation (BE) data as a complex number array.

        This property accesses the raw data from an HDF5 file. Depending on the dataset
        specified, it either retrieves the data directly from the 'Raw_Data' dataset or
        searches for a dataset that matches a noise-specific naming convention.

        Returns:
            np.array: The BE data as a complex number array.

        Example:
            data = obj.Raw_SHO_Data
            This will retrieve the raw BE data from the HDF5 file.
        """

        # Open the HDF5 file in read+write mode
        with h5py.File(self.file, "r+") as h5_f:
            # Check if the dataset is 'Raw_Data'
            if self.dataset_name == "Raw_Data":
                # Directly return the 'Raw_Data' from the HDF5 file
                name = self.dataset_name
            else:
                # If not 'Raw_Data', find the dataset that matches the noise-specific name
                name = find_measurement(
                    self.file, f"original_data_{self.noise}STD", group=self.basegroup
                )
            # Return the matched dataset
            return h5_f[f"{self.basegroup}"][name][:]

    @property
    def num_pix(self):
        """Number of pixels in the data"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f[self.measurement].attrs["num_pix"]

    @property
    def num_bins(self):
        """Number of frequency bins in the data"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f[self.measurement].attrs["num_bins"]

    @property
    def frequency_bin(self):
        """Frequency bin vector in Hz"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f[self.basegroup]["Bin_Frequencies"][:]

    @property
    def be_center_frequency(self):
        """BE center frequency in Hz"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f[self.measurement].attrs["BE_center_frequency_[Hz]"]

    @property
    def be_bandwidth(self):
        """BE bandwidth in Hz"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f[self.measurement].attrs["BE_band_width_[Hz]"]

    @property
    def be_waveform(self):
        """BE excitation waveform"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f[self.basegroup]["Excitation_Waveform"][:]

    @property
    def be_repeats(self):
        """Number of BE repeats"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f[self.measurement].attrs["BE_repeats"]

    # TODO: Josh look into this.
    @property
    def num_cycles(self):
        """
        Property to retrieve the number of cycles in the dataset.

        This method opens the HDF5 file associated with the object, reads the number of cycles
        stored in the "Measurement_000" group, and returns the total number of cycles.
        If the measurement was performed both 'in' and 'out-of-field', the number of cycles is doubled.

        Returns:
            int: The total number of cycles in the dataset.
        """

        # Open the HDF5 file in read/write mode
        with h5py.File(self.file, "r+") as h5_f:
            # Retrieve the number of cycles from the attributes of "self.measurement"
            cycles = h5_f[self.measurement].attrs["VS_number_of_cycles"]

            # JGODDY comments this out for now

            # Check if the measurement was performed 'in and out-of-field'
            # If so, double the number of cycles to account for both directions
            # if (
            #     h5_f[self.measurement].attrs["VS_measure_in_field_loops"]
            #     == "in and out-of-field"
            # ): # VS_measure_in_field_loops = 2
            #     cycles *= 2

            # Return the total number of cycles
            return cycles

    # TODO: Voltage steps should not be hardcoded.
    @property
    def dc_voltage(self):
        """Gets the DC voltage vector"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f[f"{self.raw_data_path}/Spectroscopic_Values"][0, 1::2]

    @property
    def get_voltage(self):
        """
        get_voltage gets the voltage vector

        Returns:
            np.array: voltage vector
        """

        # TODO: Look for a way to refactor and not hard code.
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f[self.basegroup]["UDVS"][::2][:, 1][24:120] * -1

    @property
    def voltage_steps(self):
        """Number of DC voltage steps"""
        with h5py.File(self.file, "r+") as h5_f:
            try:
                return h5_f[self.measurement].attrs["num_udvs_steps"]
            except:
                # computes the number of voltage steps for datasets that do not contain the attribute
                return (
                    h5_f[self.measurement].attrs["VS_steps_per_full_cycle"]
                    * h5_f[self.measurement].attrs["VS_number_of_cycles"]
                    * (
                        2
                        if h5_f[self.measurement].attrs["VS_measure_in_field_loops"]
                        == "in and out-of-field"
                        else 1
                    )
                )

    @property
    def spectroscopic_length(self):
        """Gets the length of the spectroscopic vector"""
        return self.num_bins * self.voltage_steps

    @property
    def sampling_rate(self):
        """Sampling rate in Hz"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f[self.measurement].attrs["IO_rate_[Hz]"]

    @property
    def spectroscopic_values(self):
        """Spectroscopic values"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f[self.basegroup]["Spectroscopic_Values"][:]

    @property
    def hysteresis_waveform(self, loop_number=2):
        """Gets the hysteresis waveform"""
        with h5py.File(self.file, "r+") as h5_f:
            return (
                self.spectroscopic_values[1, :: len(self.frequency_bin)][
                    int(self.voltage_steps / loop_number) :
                ]
                * self.spectroscopic_values[2, :: len(self.frequency_bin)][
                    int(self.voltage_steps / loop_number) :
                ]
            )

    @property
    def noise_std(self):
        """Gets the noise standard deviation"""
        return self.noise_std_

    @noise_std.setter
    def noise_std(self, value):
        """Sets the noise standard deviation"""

        if value is None:
            self.noise_std_ = np.std(self.raw_SHO_data)
        else:
            self.noise_std_ = value

        print(f"Noise standard deviation: {self.noise_std_}")

    # this function is very similar to get_spec_dims right below.
    # the only difference is "pos" vs "spec".
    # If I combine them it would make the code shorter but I would maybe need a way to select between the two
    # so it doesn't waste time getting the position/spectroscopic dimensions if I don't need them.
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
        self.noise_std = noise_STD

        if verbose:
            print(f"The STD of the data is: {noise_STD}")

        # Open the HDF5 file in read+write mode
        with h5py.File(self.file, "r+") as h5_f:
            # Iterate through each noise level provided in the list
            for noise_level in noise_levels:
                if (
                    usid.hdf_utils.find_dataset(h5_f, f"Noisy_Data_{noise_level}")
                    is not []
                ):
                    print(f"Noisy_Data_{noise_level} already exists")
                    continue

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
                data = self.raw_SHO_data + noise

                # Find the original dataset in the HDF5 file
                h5_main = usid.hdf_utils.find_dataset(h5_f, "Raw_Data")[0]

                # Write the noisy data to the HDF5 file
                usid.hdf_utils.write_main_dataset(
                    h5_f[self.basegroup],  # Parent group where data is saved
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

    def SHO_fit_all(self, *args, **kwargs):
        max_mem = kwargs.get("max_mem", 1024 * 64)
        max_cores = kwargs.get("max_cores", 48)

        for data in args:
            print(f"Fitting {data}")
            self.SHO_Fitter(
                dataset=data,
                h5_sho_targ_grp=f"{data}_SHO_Fit",
                max_mem=max_mem,
                max_cores=max_cores,
                **kwargs,
            )

    # this should maybe go in a separate 'preprocessing' class
    def SHO_Fitter(
        self,
        force=False,
        max_cores=-1,
        max_mem=1024 * 8,
        dataset="Raw_Data",
        h5_sho_targ_grp=None,
        return_data=False,
        SHO_fit_points=5,
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
            # JGoddy commented out the line below because I don't think either
            # data_dir or filename are used anywhere in the code.
            # (data_dir, filename) = os.path.split(self.file)

            # TODO: likeley delete.
            h5_path = self.check_H5()

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
            self.check_ckpfm(parm_dict, expt_type)

            # Handle non-BELineData types
            # if expt_type != "BELineData":
            #     vs_mode = usid.hdf_utils.get_attr(h5_meas_grp, "VS_mode")

            #     try:
            #         field_mode = usid.hdf_utils.get_attr(
            #             h5_meas_grp, "VS_measure_in_field_loops"
            #         )
            #     except KeyError:
            #         print("Field mode could not be found. Setting to default value.")
            #         field_mode = "out-of-field"

            #     try:
            #         vs_cycle_frac = usid.hdf_utils.get_attr(
            #             h5_meas_grp, "VS_cycle_fraction"
            #         )
            #     except KeyError:
            #         print(
            #             "VS cycle fraction could not be found. Setting to default value."
            #         )
            #         vs_cycle_frac = "full"

            # TODO: add a check here with a continue statement for existing SHO fits.

            # Determine the file path for saving the SHO fit results
            h5_sho_file_path = os.path.join(folder_path, h5_raw_file_name)
            print("\n\nSHO Fits will be written to:\n" + h5_sho_file_path + "\n\n")

            # Opens the file for writing or modifying
            h5_sho_file = self.upsert_to_file(h5_sho_file_path)

            # Set the target group for saving SHO results
            h5_sho_targ_grp = self.get_target_group(
                h5_sho_targ_grp, h5_file, h5_sho_file
            )

            # Initialize the SHO fitter using the specified parameters
            sho_fitter = belib.analysis.BESHOfitter(
                h5_main, cores=max_cores, verbose=False, h5_target_group=h5_sho_targ_grp
            )

            # Set up the initial guess for the SHO fitting
            sho_fitter.set_up_guess(
                guess_func=belib.analysis.be_sho_fitter.SHOGuessFunc.complex_gaussian,
                num_points=SHO_fit_points,
            )

            # Perform the initial guess fitting
            sho_fitter.do_guess(override=force)

            # Set up the actual fitting process
            sho_fitter.set_up_fit()

            # Perform the SHO fitting
            h5_sho_fit = sho_fitter.do_fit(override=force)

            # Retrieve and print the fitting parameters
            parms_dict = sidpy.hdf_utils.get_attributes(h5_main.parent.parent)
            print(
                f"LSQF method took {time.time() - start_time_lsqf} seconds to compute parameters"
            )

            # Return the fitter and fit results if requested
            if return_data:
                return sho_fitter, h5_sho_fit
            else:
                return sho_fitter

    def get_target_group(self, h5_sho_targ_grp, h5_file, h5_sho_file):
        if h5_sho_targ_grp is None:
            h5_sho_targ_grp = h5_sho_file
        else:
            h5_sho_targ_grp = make_group(h5_file, h5_sho_targ_grp)
        return h5_sho_targ_grp

    def upsert_to_file(self, h5_sho_file_path):
        f_open_mode = "w" if not os.path.exists(h5_sho_file_path) else "r+"
        h5_sho_file = h5py.File(h5_sho_file_path, mode=f_open_mode)
        return h5_sho_file

    def check_ckpfm(self, parm_dict, expt_type):
        is_ckpfm = expt_type == "cKPFMData"
        if is_ckpfm:
            num_write_steps = parm_dict["VS_num_DC_write_steps"]
            num_read_steps = parm_dict["VS_num_read_steps"]
            num_fields = 2

    def check_H5(self):
        if self.file.endswith(".h5"):
            # If the file is an HDF5 file, set the HDF5 path
            h5_path = self.file
        else:
            raise ValueError("File is not an HDF5 file")
        return h5_path

    # this function and set_SHO_LSQF are replaced by the new set_SHO_LSQF function
    # in the new code
    # I'll leave it here for now but no longer edit it
    # @context_manager_decorator
    # def set_raw_data(self):
    #     """
    #     set_raw_data Function that parses the datafile and extracts the raw data names
    #     """

    #     with h5py.File(self.file, "r+") as h5_f:
    #         # initializes the dictionary
    #         self.raw_data_reshaped = {}

    #         # list of datasets to be read
    #         datasets = []
    #         self.raw_datasets = []

    #         # Finds all the datasets
    #         datasets.extend(
    #             usid.hdf_utils.find_dataset(
    #                 h5_f["Measurement_000/Channel_000"], "Noisy"
    #             )
    #         )
    #         datasets.extend(
    #             usid.hdf_utils.find_dataset(
    #                 h5_f["Measurement_000/Channel_000"], "Raw_Data"
    #             )
    #         )

    #         # loops around all the datasets and stores them reshaped in a dictionary
    #         for dataset in datasets:
    #             self.raw_data_reshaped[dataset.name.split("/")[-1]] = dataset[
    #                 :
    #             ].reshape(self.num_pix, self.voltage_steps, self.num_bins)

    #             self.raw_datasets.extend([dataset.name.split("/")[-1]])

    # From JGoddy: I don't think we actually use this data_writer function since
    # I never uncommented it but I'm putting it here for now (still uncommented)

    # def data_writer(self, base, name, data):
    #     """
    #     data_writer function to write data to an USID dataset

    #     Args:
    #         base (str): basepath where to save the data
    #         name (str): name of the dataset to save
    #         data (np.array): data to save
    #     """

    #     with h5py.File(self.file, "r+") as h5_f:

    #         try:
    #             # if the dataset does not exist can write
    #             make_dataset(h5_f[base],
    #                          name,
    #                          data)

    #         except:
    #             # if the dataset exists deletes the dataset and then writes
    #             self.delete(f"{base}/{name}")
    #             make_dataset(h5_f[base],
    #                          name,
    #                          data)

    # this function replaces:
    # set_SHO_LSQF (self.SHO_LSQF_data)
    # set_raw_data (self.raw_data_reshaped)
    def set_SHO_LSQF(self):
        """
        set_SHO_LSQF Sets the SHO Scaler data to make accessible
        """

        # initializes the dictionary
        self.SHO_LSQF_data = {}
        self.raw_data_reshaped = {}

        with h5py.File(self.file, "r+") as h5_f:
            self.SHO_LSQF_data[self.dataset_name] = structured_to_unstructured(
                h5_f[f"{self.dataset_name}-{self.SHO_fit_relative_base_path}/Fit"][:]
            )[:, :, :-1]

            self.raw_data_reshaped[self.dataset_name] = h5_f[
                f"{self.basegroup}/{self.dataset_name}"
            ][:].reshape(self.num_pix, self.voltage_steps, self.num_bins)

        # for dataset in self.raw_datasets:
        #     # data groups in file
        #     try:
        #         SHO_fits = find_groups_with_string(self.file, f"{dataset}-SHO_Fit_000")[0]

        #         with h5py.File(self.file, "r+") as h5_f:
        #             # extract the name of the fit
        #             name = SHO_fits.split("/")[-1]

        #             # create a list for parameters
        #             SHO_LSQF_list = []
        #             for sublist in np.array(h5_f[f"{SHO_fits}/Fit"]):
        #                 for item in sublist:
        #                     for i in item:
        #                         SHO_LSQF_list.append(i)

        #             data_ = np.array(SHO_LSQF_list).reshape(-1, 5)

        #             # saves the SHO LSQF data as an attribute of the dataset object
        #             self.SHO_LSQF_data[name] = data_.reshape(
        #                 self.num_pix, self.voltage_steps, 5
        #             )[:, :, :-1]
        #     except Exception as e:
        #         if isinstance(e, IndexError):
        #             print("*"*20)
        #             print(f"SHO_LSQF_data for {dataset} not found")
        #             print("Skipping retrieval of SHO_LSQF_data for this dataset")
        #             print("*"*20)
        #         else:
        #             print("set_SHO_LSQF failed with exception:")
        #             print(e)
        #             print("*"*20)
        #             print("Traceback:")
        #             print(traceback.format_exc())

    # JGoddy put this function here because it relates the the h5 files
    # but it doesn't actually use the h5 file so maybe it should be elsewhere?
    def get_loop_path(self):
        """
        get_loop_path gets the path where the hysteresis loops are located

        Returns:
            str: string pointing to the path where the hysteresis loops are located
        """

        if self.noise == 0 or self.noise is None:
            prefix = "Raw_Data"
            return f"{self.measurement}/{prefix}-{self.SHO_fit_relative_base_path}/{self.SHO_hysteresis_loop_fit_name}"
        else:
            prefix = f"Noisy_Data_{self.noise}"
            return f"/Noisy_Data_{self.noise}_SHO_Fit/Noisy_Data_{self.noise}-{self.SHO_fit_relative_base_path}/{self.SHO_hysteresis_loop_guess_name}"

    @static_state_decorator
    def get_hysteresis(
        self,
        fits=False,
        noise=None,
        plotting_values=False,
        output_shape=None,
        scaled=None,
        loop_interpolated=None,
        measurement_state=None,
    ):
        """
        get_hysteresis function to get the hysteresis loops

        Args:
            noise (int, optional): sets the noise value. Defaults to None.
            plotting_values (bool, optional): sets if you get the data shaped for computation or plotting. Defaults to False.
            output_shape (str, optional): sets the shape of the output. Defaults to None.
            scaled (any, optional): selects if the output is scaled or unscaled. Defaults to None.
            loop_interpolated (any, optional): sets if you should get the interpolated loops. Defaults to None.
            measurement_state (any, optional): sets the measurement state. Defaults to None.

        Returns:
            np.array: output hysteresis data, bias vector for the hysteresis loop
        """

        # todo: can replace this to make this much nicer to get the data. Too many random transforms

        if measurement_state is not None:
            self.measurement_state = measurement_state

        with h5py.File(self.file, "r+") as h5_f:
            # sets the noise value
            if noise is None:
                self.noise = noise

            # sets the output shape
            if output_shape is not None:
                self.output_shape = output_shape

            # selects if the scaled data is returned
            if scaled is not None:
                self.scaled = scaled

            # selects if interpolated hysteresis loops are returned
            if loop_interpolated is not None:
                self.loop_interpolated = loop_interpolated

            # gets the path where the hysteresis loops are located
            h5_path = self.get_loop_path()

            if fits is False:
                # gets the projected loops
                h5_projected_loops = h5_f[h5_path + "/Projected_Loops"]
            else:
                h5_projected_loops = h5_f[h5_path + "/Fit"]

            # Prepare some variables for plotting loops fits and guesses
            # Plot the Loop Guess and Fit Results
            proj_nd, _ = reshape_to_n_dims(h5_projected_loops)

            spec_ind = get_auxiliary_datasets(
                h5_projected_loops, aux_dset_name="Spectroscopic_Indices"
            )[-1]
            spec_values = get_auxiliary_datasets(
                h5_projected_loops, aux_dset_name="Spectroscopic_Values"
            )[-1]
            pos_ind = get_auxiliary_datasets(
                h5_projected_loops, aux_dset_name="Position_Indices"
            )[-1]

            pos_nd, _ = reshape_to_n_dims(pos_ind, h5_pos=pos_ind)
            pos_dims = list(pos_nd.shape[: pos_ind.shape[1]])

            # reshape the vdc_vec into DC_step by Loop
            spec_nd, _ = reshape_to_n_dims(spec_values, h5_spec=spec_ind)
            loop_spec_dims = np.array(spec_nd.shape[1:])
            loop_spec_labels = sidpy.hdf.hdf_utils.get_attr(spec_values, "labels")

            spec_step_dim_ind = np.where(loop_spec_labels == "DC_Offset")[0][0]

            # Also reshape the projected loops to Positions-DC_Step-Loop
            final_loop_shape = pos_dims + [loop_spec_dims[spec_step_dim_ind]] + [-1]
            proj_nd2 = np.moveaxis(
                proj_nd, spec_step_dim_ind + len(pos_dims), len(pos_dims)
            )
            proj_nd_3 = np.reshape(proj_nd2, final_loop_shape)

            # Get the bias vector:
            spec_nd2 = np.moveaxis(spec_nd[spec_step_dim_ind], spec_step_dim_ind, 0)
            bias_vec = np.reshape(spec_nd2, final_loop_shape[len(pos_dims) :])

            if plotting_values:
                proj_nd_3, bias_vec = self.roll_hysteresis(bias_vec, proj_nd_3)

            hysteresis_data = np.transpose(proj_nd_3, (1, 0, 3, 2))

            # interpolates the data
            if self.loop_interpolated:
                hysteresis_data = clean_interpolate(hysteresis_data)

            # transforms the data with the scaler if necessary.
            if self.scaled:
                hysteresis_data = self.hysteresis_scaler_.transform(hysteresis_data)

            # sets the data to the correct output shape
            if self.output_shape == "index":
                hysteresis_data = proj_nd_3.reshape(
                    self.num_cycles * self.num_pix,
                    self.voltage_steps // self.num_cycles,
                )
            elif self.output_shape == "pixels":
                pass

            hysteresis_data = self.hysteresis_measurement_state(hysteresis_data)

        # output shape (x,y, cycle, voltage_steps)
        # bias_vec
        return hysteresis_data, np.swapaxes(
            np.atleast_2d(self.get_voltage), 0, 1
        ).astype(np.float64)

    # JGoddy doesn't know if the following function should go here
    # but putting it here for now because it is used by the
    # LSQF_Loop_Fit function below

    def measure_group(self):
        """
        measure_group gets the measurement group based on a noise level

        Returns:
            str: string for the measurement group for the data
        """

        if self.noise == 0:
            return "Raw_Data_SHO_Fit"
        else:
            return f"Noisy_Data_{self.noise}"

    def LSQF_Loop_Fit(
        self,
        main_dataset=None,
        h5_target_group=None,
        max_cores=None,
        force=False,
        h5_sho_targ_grp=None,
    ):
        """
        LSQF_Loop_Fit Function that conducts the hysteresis loop fits based on the LSQF results.

        This is adapted from BGlib

        Args:
            main_dataset (str, optional): main dataset where loop fits are conducted from. Defaults to None.
            h5_target_group (str, optional): path where the data will be saved to. Defaults to None.
            max_cores (int, optional): number of cores the fitter will use, -1 will use all cores. Defaults to None.
            h5_sho_targ_grp (str, optional): path where the SHO fits are saved. Defaults to None.

        Raises:
            TypeError: _description_

        Returns:
            tuple: results from the loop fit, group where the loop fit is
        """

        with h5py.File(self.file, "r+") as h5_file:
            # finds the main dataset location in the file
            h5_main = self.get_main_dataset(main_dataset, h5_file)

            # gets the measurement group name
            h5_meas_grp = h5_main.parent.parent

            # does the SHO_fit if it does not exist.
            sho_fit_points = (
                5  # The number of data points at each step to use when fitting
            )
            sho_override = False  # Force recompute if True
            sho_fitter = belib.analysis.BESHOfitter(
                h5_main, cores=max_cores, verbose=False, h5_target_group=h5_meas_grp
            )
            sho_fitter.set_up_guess(
                guess_func=belib.analysis.be_sho_fitter.SHOGuessFunc.complex_gaussian,
                num_points=sho_fit_points,
            )
            h5_sho_guess = sho_fitter.do_guess(override=sho_override)
            sho_fitter.set_up_fit()
            h5_sho_fit = sho_fitter.do_fit(override=sho_override)
            h5_sho_grp = h5_sho_fit.parent

            # gets the experiment type from the file
            expt_type = sidpy.hdf.hdf_utils.get_attr(h5_file, "data_type")

            # finds the dataset from the file
            h5_meas_grp = usid.hdf_utils.find_dataset(h5_file, self.measure_group())

            # extract the voltage mode
            vs_mode = sidpy.hdf.hdf_utils.get_attr(
                h5_file["/Measurement_000"], "VS_mode"
            )

            try:
                vs_cycle_frac = sidpy.hdf.hdf_utils.get_attr(
                    h5_file["/Measurement_000"], "VS_cycle_fraction"
                )

            except KeyError:
                print("VS cycle fraction could not be found. Setting to default value")
                vs_cycle_frac = "full"

            sho_fit, sho_dataset = self.SHO_Fitter(return_data=True)

            # instantiates the loop fitter using belib
            loop_fitter = belib.analysis.BELoopFitter(
                h5_sho_fit,
                expt_type,
                vs_mode,
                vs_cycle_frac,
                #  h5_target_group=h5_meas_grp,
                cores=max_cores,
                verbose=False,
            )

            # computes the guess for the loop fits
            loop_fitter.set_up_guess()
            h5_loop_guess = loop_fitter.do_guess(override=force)

            # Calling explicitly here since Fitter won't do it automatically
            h5_guess_loop_parms = loop_fitter.extract_loop_parameters(h5_loop_guess)
            loop_fitter.set_up_fit()
            h5_loop_fit = loop_fitter.do_fit(override=force)

            # save the path where the loop fit results are saved
            h5_loop_group = h5_loop_fit.parent

        return h5_loop_fit, h5_loop_group

    def get_main_dataset(self, main_dataset, h5_file):
        if main_dataset is None:
            h5_main = usid.hdf_utils.find_dataset(h5_file, "Raw_Data")[0]
        else:
            h5_main = usid.hdf_utils.find_dataset(h5_file, main_dataset)[0]
        return h5_main

    @static_state_decorator
    def LSQF_hysteresis_params(
        self, output_shape=None, scaled=None, measurement_state=None
    ):
        """
        LSQF_hysteresis_params Gets the LSQF hysteresis parameters

        Args:
            output_shape (str, optional): pixel or list. Defaults to None.
            scaled (bool, optional): selects if to scale the data. Defaults to None.
            measurement_state (any, optional): sets the measurement state. Defaults to None.

        Returns:
            np.array: hysteresis loop parameters from LSQF
        """

        if measurement_state is not None:
            self.measurement_state = measurement_state

        # sets output shape if provided
        if output_shape is not None:
            self.output_shape = output_shape

        # sets data to be scaled is provided
        if scaled is not None:
            self.scaled = scaled

        # extracts the hysteresis parameters from the H5 file
        with h5py.File(self.file, "r+") as h5_f:
            # data = h5_f[f"/Measurement_000/{self.dataset}-SHO_Fit_000/Fit-Loop_Fit_000/Fit"][:]
            data = h5_f[
                f"/{self.measurement}/{self.dataset_name}-{self.SHO_fit_relative_base_path}/{self.SHO_hysteresis_relative_base_path}/Fit"
            ][:]
            data = data.reshape(self.num_rows, self.num_cols, self.num_cycles)
            data = np.array(
                [
                    data["a_0"],
                    data["a_1"],
                    data["a_2"],
                    data["a_3"],
                    data["a_4"],
                    data["b_0"],
                    data["b_1"],
                    data["b_2"],
                    data["b_3"],
                ]
            ).transpose((1, 2, 3, 0))

            if self.scaled:
                # TODO: add the scaling here
                data = self.loop_param_scaler.fit(data)

                # Warning("Scaling not implemented yet")
                # pass

            if self.output_shape == "index":
                data = data.reshape(self.num_pix, self.num_cycles, data.shape[-1])

            data = self.hysteresis_measurement_state(data)

            return data
