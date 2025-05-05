import os
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

import time
import sidpy
from BGlib import be as belib

from m3util.util.h5 import (
    print_tree,
    get_tree,
    find_measurement,
    make_group,
    find_groups_with_string,
)
import pyUSID as usid

import h5py
from pyUSID.io.hdf_utils import reshape_to_n_dims, get_auxiliary_datasets


from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path
from belearn.util.wrappers import context_manager_decorator
from belearn.filters.filters import clean_interpolate

from typing import Dict, List, Tuple, Any


# TODO: Move Fitting to a separate class, SHO and Hysteresis Loop
@dataclass
class BE_Dataset:
    """
    A class to represent a BE (Band Excitation) dataset stored in an HDF5 file.

    This class provides attributes and methods to interact with and manipulate
    the data stored in the HDF5 file, including handling noise levels and
    resampling data.

    Attributes:
        file (str): The path to the HDF5 file.
        noise (int): The noise level to be applied to the dataset. Defaults to 0.
        resampled_bins (int): The number of bins to resample the data to. Defaults to None.
        resampled_data (dict): A dictionary containing the resampled data. Defaults to None.
        datafed (Optional[Union[None, str, Path]]): An optional path or identifier for data federation. Defaults to None.
        basegroup (str): The base group path in the HDF5 file. Defaults to "/Measurement_000/Channel_000".
        raw_data_path (str): The path to the raw data within the HDF5 file. Defaults to "Raw_Data_SHO_Fit/Raw_Data-SHO_Fit_000".
        measurement_data_path (str): The path to the measurement data within the HDF5 file. Defaults to "Measurement_Data/Measurement_Data-000".
        measurement (str): The measurement identifier within the HDF5 file. Defaults to "Measurement_000".
        SHO_fit_relative_base_path (str): The relative path for SHO fit data. Defaults to "SHO_Fit_000".
        SHO_hysteresis_loop_fit_name (str): The name for the SHO hysteresis loop fit. Defaults to "Fit-Loop_Fit_000".
        SHO_hysteresis_loop_guess_name (str): The name for the SHO hysteresis loop guess. Defaults to "Guess-Loop_Fit_000".
        noise_std_ (float): The standard deviation of the noise. Defaults to None.

    Methods:
        get_dataset(noise):
            Returns the current dataset based on the noise state.

        tree:
            Reads the tree from the H5 file and returns it as a list.

        print_be_tree:
            Prints the Band Excitation tree structure from the H5 file.

        num_pix:
            Returns the number of pixels in the data.

        num_bins:
            Returns the number of frequency bins in the data.

        frequency_bin:
            Returns the frequency bin vector in Hz.

        be_center_frequency:
            Returns the BE center frequency in Hz.

        be_bandwidth:
            Returns the BE bandwidth in Hz.

        be_waveform:
            Returns the BE excitation waveform.

        be_repeats:
            Returns the number of BE repeats.

        num_cycles:
            Retrieves the number of cycles in the dataset.

        dc_voltage:
            Gets the DC voltage vector.

        get_voltage:
            Gets the voltage vector.

        voltage_steps:
            Returns the number of voltage steps.

        measure_group:
            Gets the measurement group based on a noise level.

        LSQF_Loop_Fit(main_dataset, h5_target_group, max_cores, force, h5_sho_targ_grp):
            Conducts the hysteresis loop fits based on the LSQF results.

        get_main_dataset(main_dataset, h5_file):
            Finds the main dataset location in the file.

        LSQF_hysteresis_params(output_shape, scaled, measurement_state):
            Gets the LSQF hysteresis parameters.

        SHO_fit_all(*args, **kwargs):
            Fits the Simple Harmonic Oscillator (SHO) model to all provided datasets.

        SHO_Fitter(force, max_cores, max_mem, dataset, h5_sho_targ_grp, return_data, SHO_fit_points):
            Computes the SHO fit results for a given dataset.
    """

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

    def __post_init__(self):
        """
        Post-initialization method for the BE_Dataset class.

        This method is automatically called after the class is initialized. It sets the
        data federation attribute and determines the current dataset based on the noise level.
        """
        self.get_dataset(self.noise)

        # TODO: remove this
        # The following lines are commented out as they are not currently in use.
        # They are intended for initializing resampled_bins and resampled_data attributes.
        # self.resampled_bins = self.resampled_bins
        # self.resampled_data = self.resampled_data
        # # Initialize resampled_bins if it's None
        # if self.resampled_bins is None:
        #     self.resampled_bins = self.num_bins

    def get_dataset(self, noise: int):
        """
        Determines the current dataset name based on the noise level.

        This method sets the `dataset_name` attribute to either "Raw_Data"
        if the noise level is zero, or to a noise-specific dataset name
        formatted as "Noisy_Data_{noise}" for non-zero noise levels.

        Args:
            noise (int): The noise level used to determine the dataset name.
        """
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
                "Datasets and data groups within the file:\n------------------------------------"
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
                "\nMetadata or attributes in a data group\n------------------------------------"
            )

            for key in h5_f.file[self.measurement].attrs:
                print("{} : {}".format(key, h5_f.file[self.measurement].attrs[key]))

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
            # I need the number of cycles to be 4 not 2 (so the stuff below) 
            # for the hysteresis model not I need it to be 2 for the SHO model
            # so I think I will just change it elsewhere (see LSQF_hysteresis_params)

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
            # TODO: Fix hardcoded values.
            return h5_f[self.basegroup]["UDVS"][::2][:, 1][24:120] * -1

    @property
    def voltage_steps(self):
        """Number of DC voltage steps"""
        with h5py.File(self.file, "r+") as h5_f:
            try:
                return h5_f[self.measurement].attrs["num_udvs_steps"]
            except:  # noqa: E722
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
        return (
            self.spectroscopic_values[1, :: len(self.frequency_bin)][
                int(self.voltage_steps / loop_number) :
            ]
            * self.spectroscopic_values[2, :: len(self.frequency_bin)][
                int(self.voltage_steps / loop_number) :
            ]
        )

    @property
    def num_cols(self):
        """Number of columns in the data"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f['Measurement_000'].attrs["grid_num_cols"]

    @property
    def num_rows(self):
        """Number of rows in the data"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f['Measurement_000'].attrs["grid_num_rows"]
        
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
        noise_levels: list[float],
        verbose: bool = False,
        noise_STD: float | None = None,
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
            print(f"The STD of the data is: {self.noise_std}")

        # Open the HDF5 file in read+write mode
        with h5py.File(self.file, "r+") as h5_f:
            # Iterate through each noise level provided in the list
            for noise_level in noise_levels:
                if usid.hdf_utils.find_dataset(h5_f, f"Noisy_Data_{noise_level}") != []:
                    print(f"Noisy_Data_{noise_level} already exists")
                    continue

                if verbose:
                    print(f"Adding noise level {noise_level}")

                # Calculate the actual noise level to be applied
                noise_level_ = self.noise_std * noise_level

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

    # TODO: move to SHOFitter class
    def SHO_fit_all(self, *args: Any, **kwargs: Any):
        """
        Fits the Simple Harmonic Oscillator (SHO) model to all provided datasets.

        This method iterates over each dataset provided in the arguments and applies
        the SHO fitting process using the specified memory and core constraints.

        Args:
            *args: Variable length argument list containing datasets to be fitted.
            **kwargs: Arbitrary keyword arguments. Supported keys include:
                - max_mem (int): Maximum memory in MB to be used for fitting. Defaults to 65536 MB.
                - max_cores (int): Maximum number of CPU cores to be used for fitting. Defaults to 48.

        Example:
            >>> obj.SHO_fit_all("dataset1", "dataset2", max_mem=32768, max_cores=24)

        """
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

    def SHO_Fitter(
        self,
        force: bool = False,
        max_cores: int = -1,
        max_mem: int = 1024 * 8,
        dataset: str = "Raw_Data",
        h5_sho_targ_grp: h5py.Group | None = None,
        return_data: bool = False,
        SHO_fit_points: int = 5,
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

            h5_path = self.check_H5()

            # Split the path to get the folder and raw file name
            folder_path, h5_raw_file_name = os.path.split(h5_path)

            print("Working on:\n" + h5_path)

            # Get the main dataset to be fitted
            h5_main = usid.hdf_utils.find_dataset(h5_file, dataset)[0]

            # Extract useful parameters from the dataset
            pos_ind = h5_main.h5_pos_inds  # noqa: F841
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

            # TODO: JGoddy doesn't remember why this code in commented out
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

            if (
                self.get_data_group_contents(search_string = dataset) is not None
                and force is False
                and return_data is False
            ):
                print(f"SHO fits for {dataset} already exist. Skipping....")

            else:
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
                parameter_dict = sidpy.hdf_utils.get_attributes(h5_main.parent.parent)  # noqa: F841
                print(
                    f"LSQF method took {time.time() - start_time_lsqf} seconds to compute parameters"
                )

            # Return the fitter and fit results if requested
            if return_data:
                return sho_fitter, h5_sho_fit
            else:
                return sho_fitter

    def get_data_group_contents(self, search_string: str="Raw_Data"):
        """
        Retrieves the contents of the first data group containing 'Raw_Data' in its name.

        This method searches for groups within the HDF5 file that contain the string 'Raw_Data'
        in their names. It then opens the file and retrieves the keys of the first matching group.
        
        Args:
            search_string (str): The string to search for in the group names.

        Returns:
            list or None: A list of keys within the first matching data group if it exists,
            otherwise None if the group is empty or not found.
        """
        datagroup = find_groups_with_string(self.file, search_string)

        with h5py.File(self.file, "r+") as h5_f:
            data_group_contents = h5_f[datagroup[0]].keys()

            if not data_group_contents:
                return None
            else:
                return data_group_contents

    def get_target_group(
        self,
        h5_sho_targ_grp: h5py.Group | None,
        h5_file: h5py.File,
        h5_sho_file: h5py.File,
    ):
        """
        Determines the target HDF5 group for saving SHO results.

        This function checks if a target group is provided. If not, it defaults to using
        the provided HDF5 file as the target group. If a target group is specified, it
        ensures the group exists within the HDF5 file, creating it if necessary.

        Args:
            h5_sho_targ_grp (h5py.Group | None): The target group for saving SHO results.
                If None, the function defaults to using the provided HDF5 file.
            h5_file (h5py.File): The HDF5 file where the group should be located or created.
            h5_sho_file (h5py.File): The HDF5 file to use as the default target group if
                no specific group is provided.

        Returns:
            h5py.Group: The determined target group for saving SHO results.
        """
        if h5_sho_targ_grp is None:
            h5_sho_targ_grp = h5_sho_file
        else:
            h5_sho_targ_grp = make_group(h5_file, h5_sho_targ_grp)
        return h5_sho_targ_grp

    def upsert_to_file(self, h5_sho_file_path: str) -> h5py.File:
        """
        Opens an HDF5 file for reading and writing, creating it if it does not exist.

        This method checks if the specified HDF5 file exists at the given path. If the file
        does not exist, it opens the file in write mode to create it. If the file already exists,
        it opens the file in read and write mode.

        Args:
            h5_sho_file_path (str): The file path to the HDF5 file to be opened or created.

        Returns:
            h5py.File: An HDF5 file object opened in the appropriate mode.
        """
        f_open_mode = "w" if not os.path.exists(h5_sho_file_path) else "r+"
        h5_sho_file = h5py.File(h5_sho_file_path, mode=f_open_mode)
        return h5_sho_file

    def check_ckpfm(self, parm_dict: dict[str, Any], expt_type: str) -> bool:
        """
        Checks if the experiment type is cKPFMData and retrieves relevant parameters.

        This function determines whether the provided experiment type corresponds to
        cKPFMData. If it does, it extracts specific parameters from the provided
        parameter dictionary, such as the number of DC write steps, read steps, and
        fields, which are used in the cKPFM experiment.

        Args:
            parm_dict (dict[str, Any]): A dictionary containing parameters for the experiment.
            expt_type (str): The type of experiment being conducted.

        Returns:
            bool: True if the experiment type is cKPFMData, False otherwise.
        """
        is_ckpfm = expt_type == "cKPFMData"

        if is_ckpfm:
            num_write_steps = parm_dict["VS_num_DC_write_steps"]  # noqa: F841
            num_read_steps = parm_dict["VS_num_read_steps"]  # noqa: F841
            num_fields = 2  # noqa: F841
        return is_ckpfm

    def check_H5(self) -> str:
        """
        Validates if the file attribute is an HDF5 file and returns its path.

        This method checks if the file associated with the BE_Dataset instance
        has an '.h5' extension, indicating it is an HDF5 file. If the file is
        valid, it returns the file path. Otherwise, it raises a ValueError.

        Returns:
            str: The path to the HDF5 file.

        Raises:
            ValueError: If the file does not have an '.h5' extension.
        """
        if self.file.endswith(".h5"):
            return self.file
        else:
            raise ValueError("File is not an HDF5 file")

    def set_SHO_LSQF(self):
        """
        Initializes and sets the Simple Harmonic Oscillator (SHO) Scaler data for accessibility.

        This method prepares the SHO Scaler data by initializing necessary dictionaries
        and reshaping raw data for further analysis. It reads the data from an HDF5 file
        and stores it in a structured format for easy access.

        The method performs the following actions:
        - Initializes `SHO_LSQF_data` and `raw_data_reshaped` dictionaries.
        - Reads and processes the SHO fit data from the HDF5 file.
        - Reshapes the raw data for analysis.

        Raises:
            KeyError: If the specified dataset or path does not exist in the HDF5 file.
        """

        # Initialize the dictionaries for storing data
        self.SHO_LSQF_data = {}
        self.raw_data_reshaped = {}

        print(
            f"Accessing data at: {self.dataset_name}-{self.SHO_fit_relative_base_path}/Fit"
        )

        with h5py.File(self.file, "r+") as h5_f:
            try:
                # Extract and store the SHO fit data
                self.SHO_LSQF_data[self.dataset_name] = structured_to_unstructured(
                    h5_f[f"{self.dataset_name}-{self.SHO_fit_relative_base_path}/Fit"][
                        :
                    ]
                )[:, :, :-1]

                # Reshape and store the raw data
                self.raw_data_reshaped[self.dataset_name] = h5_f[
                    f"{self.basegroup}/{self.dataset_name}"
                ][:].reshape(self.num_pix, self.voltage_steps, self.num_bins)
            except KeyError as e:
                raise KeyError(f"Dataset or path not found in HDF5 file: {e}")

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

    @context_manager_decorator
    def get_hysteresis(
        self,
        fits: Optional[bool] = False,
        noise: Optional[int] = None,
        plotting_values: Optional[bool] = False,
        output_shape: Optional[str] = None,
        scaled: Optional[Any] = None,
        loop_interpolated: Optional[Any] = None,
        measurement_state: Optional[Any] = None,
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

    def get_measure_group_name(self) -> str:
        """
        Retrieves the measurement group name based on the current noise level.

        This method determines the appropriate measurement group name for the dataset
        by checking the noise level. If the noise level is zero, it returns the group
        name for raw data. Otherwise, it returns the group name for noisy data
        corresponding to the specified noise level.

        Returns:
            str: The measurement group name for the dataset.
        """
        if self.noise == 0:
            return "Raw_Data_SHO_Fit"
        else:
            return f"Noisy_Data_{self.noise}"

    # TODO: Refactor and update.
    def LSQF_Loop_Fit(
        self,
        main_dataset: Optional[str] = None,
        h5_target_group: Optional[str] = None,
        max_cores: Optional[int] = None,
        force: Optional[bool] = False,
        h5_sho_targ_grp: Optional[str] = None,
        SHO_fit_points: Optional[int] = 5,
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

            
            sho_override = False  # Force recompute if True
            sho_fitter = belib.analysis.BESHOfitter(
                h5_main, cores=max_cores, verbose=False, h5_target_group=h5_meas_grp
            )
            sho_fitter.set_up_guess(
                guess_func=belib.analysis.be_sho_fitter.SHOGuessFunc.complex_gaussian,
                num_points=SHO_fit_points,
            )
            h5_sho_guess = sho_fitter.do_guess(override=sho_override)  # noqa: F841
            sho_fitter.set_up_fit()
            h5_sho_fit = sho_fitter.do_fit(override=sho_override)
            h5_sho_grp = h5_sho_fit.parent  # noqa: F841

            # gets the experiment type from the file
            expt_type = sidpy.hdf.hdf_utils.get_attr(h5_file, "data_type")

            # finds the dataset from the file
            h5_meas_grp = usid.hdf_utils.find_dataset(
                h5_file, self.get_measure_group_name()
            )

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
            h5_guess_loop_parms = loop_fitter.extract_loop_parameters(h5_loop_guess)  # noqa: F841
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

    @context_manager_decorator
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
                f"/{self.measurement}/{self.dataset_name}-{self.SHO_fit_relative_base_path}/{self.SHO_hysteresis_loop_fit_name}/Fit"
            ][:]
            data = data.reshape(self.num_rows, self.num_cols, 2*self.num_cycles)
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
