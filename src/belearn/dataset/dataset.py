from m3util.util.search import in_list
from m3util.util.h5 import (
    find_groups_with_string,
    print_tree,
    get_tree,
    find_measurement,
    make_group,
)
from belearn.dataset.scalers import Raw_Data_Scaler
from belearn.util.wrappers import static_state_decorator
from belearn.functions.sho import SHO_nn
from belearn.functions.hysteresis import hysteresis_nn
from belearn.filters.filters import clean_interpolate
from m3util.ml.preprocessor import GlobalScaler

# from belearn.dataset.transformers import to_complex
import numpy as np
from dataclasses import dataclass, field, InitVar
import h5py
import time
import sidpy
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
import pyUSID as usid
from scipy.signal import resample
from typing import Any, Callable, Dict, Optional
from BGlib import be as belib
import torch
from typing import Optional, Union
from pathlib import Path
from datafed_torchflow.datafed import DataFed
from m3util.util.hashing import calculate_h5file_checksum
from pyUSID.io.hdf_utils import reshape_to_n_dims, get_auxiliary_datasets


@dataclass
class BE_Dataset:
    file: str
    scaled: bool = False
    raw_format: str = "complex"
    fitter: str = "LSQF"
    output_shape: str = "pixels"
    measurement_state: str = "all"
    resampled: bool = False
    resampled_bins: Optional[int] = field(default=None, init=False)
    datafed: Optional[Union[None, str, Path]] = None
    LSQF_phase_shift: Optional[float] = None
    NN_phase_shift: Optional[float] = None
    verbose: bool = False
    noise_state: int = 0
    cleaned: bool = False
    basegroup: str = "/Measurement_000/Channel_000"
    SHO_fit_func_LSQF: Callable = field(default=SHO_nn)
    hysteresis_function: Optional[Callable] = hysteresis_nn
    loop_interpolated: bool = False
    tree: Any = field(init=False)
    resampled_data: Dict[str, Any] = field(default_factory=dict, init=False)
    dataset_id: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    """A class to represent a Band Excitation (BE) dataset.

    Attributes:
        file (str): The file path of the dataset.
        scaled (bool, optional): Whether the data is scaled. Defaults to False.
        raw_format (str, optional): The raw data format. Defaults to "complex".
        fitter (str, optional): The fitter to be used. Defaults to 'LSQF'.
        output_shape (str, optional): The output shape of the dataset. pixels is 2d, index is 1d Defaults to 'pixels'.
        measurement_state (str, optional): The state of the measurement. Defaults to 'all'.
        resampled (bool, optional): Whether the data is resampled. Defaults to False.
        resampled_bins (int, optional): The number of bins for resampling. Automatically set if None.
        LSQF_phase_shift (float, optional): The phase shift for LSQF.
        NN_phase_shift (float, optional): The phase shift for Neural Network.
        verbose (bool, optional): Whether to print detailed information. Defaults to False.
        noise_state (int, optional): Noise level. Defaults to 0.
        cleaned (bool, optional): Whether the data is cleaned. Defaults to False.
        basegroup (str, optional): The base group in the HDF5 file. Defaults to '/Measurement_000/Channel_000'.
        SHO_fit_func_LSQF (Callable, optional): The fitting function for SHO in NN.
        hysteresis_function (Callable, optional): The hysteresis function for processing. 
        loop_interpolated (bool, optional): Whether the loop data is interpolated. Defaults to False.
        resampled_data (Dict[str, Any]): Holds resampled data. Initialized post object creation.
        dataset_id (str, optional): The DataFed ID of the dataset. Defaults to None.
        kwargs (Dict[str, Any], optional): Additional keyword arguments.
    """

    def __post_init__(self):
        self.noise = self.noise_state
        self.tree = self.get_tree()

        # Initialize resampled_bins if it's None
        if self.resampled_bins is None:
            self.resampled_bins = self.num_bins

        # Instantiate DataFed object
        self.instantiate_datafed()

        # Set additional attributes from kwargs
        for key, value in self.kwargs.items():
            setattr(self, key, value)

        # Preprocessing and raw data
        self.set_preprocessing()
        self.set_raw_data()
        self.SHO_preprocessing()

    def set_preprocessing(self):
        """
        set_preprocessing searches the dataset to see what preprocessing is required.
        """

        # does preprocessing for the SHO_fit results
        if in_list(self.tree, "*SHO_Fit*"):
            self.SHO_preprocessing()
        else:
            Warning("No SHO fit found")

        # does preprocessing for the loop fit results
        if in_list(self.tree, "*Fit-Loop_Fit*"):
            self.loop_fit_preprocessing()

    def SHO_preprocessing(self):
        """
        SHO_preprocessing conducts the preprocessing on the SHO fit results
        """

        # extract the raw data and reshapes is
        self.set_raw_data()

        # # resamples the data if necessary
        self.set_raw_data_resampler()

        # computes the scalar on the raw data
        self.raw_data_scaler = Raw_Data_Scaler(self.raw_data())

        try:
            # gets the LSQF results
            self.set_SHO_LSQF()

            # computes the SHO scaler
            self.SHO_Scaler()
        except:
            pass

    def set_SHO_LSQF(self):
        """
        set_SHO_LSQF Sets the SHO Scaler data to make accessible
        """

        # initializes the dictionary
        self.SHO_LSQF_data = {}

        for dataset in self.raw_datasets:
            # data groups in file
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

    ##### DATAFED ###

    def instantiate_datafed(self):
        """
        Instantiate the DataFed object and upload the dataset to DataFed if applicable.

        This method checks if the `datafed` attribute is provided and valid. If it starts with "d/",
        it is assumed to be a DataFed ID. Otherwise, it attempts to create a DataFed object and upload
        the dataset to DataFed, extracting metadata and handling the upload process.

        Raises:
            ValueError: If the `datafed` attribute is not a valid DataFed path or identifier.
        """
        if self.datafed is not None and self.datafed.startswith("d/"):
            # If datafed is a DataFed ID, set it directly
            self.dataset_id = self.datafed
        elif self.datafed is not None:
            # Instantiate the DataFed object if datafed is a path
            self.datafed_obj = DataFed(self.datafed)

            # Extract metadata from the HDF5 file structure
            metadata = self.extract_h5_structure()

            metadata.update(
                {
                    "checksum": calculate_h5file_checksum(self.file),
                }
            )

            # Create a data record in DataFed with the extracted metadata
            dc_resp = self.datafed_obj.data_record_create(metadata, self.file.split("/")[-1].split(".")[0])

            # Upload the file to DataFed
            self.datafed_obj.upload_file(dc_resp, self.file, wait=False)

            # Set the DataFed ID from the response
            self.dataset_id = dc_resp[0].data[0].id

        elif self.datafed is None:
            # If datafed is None, set dataset_id to None
            self.dataset_id = None
        else:
            # Raise an error if datafed is not a valid path or identifier
            raise ValueError("DataFed value is not a valid DataFed path or identifier")

    @static_state_decorator
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
            noise_STD = np.std(self.get_original_data)

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
                data = self.get_original_data + noise

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

    @staticmethod
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
        if BE_Dataset.is_complex(data):
            return data

        # if axis is not provided take the last axis
        if axis is None:
            axis = data.ndim - 1

        return np.take(data, 0, axis=axis) + 1j * np.take(data, 1, axis=axis)

    @staticmethod
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

    ##### SHO FITTERS #####

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

    def extract_h5_structure(self) -> dict:
        """
        Extracts the structure, main dataset, ancillary datasets, and metadata from an HDF5 file
        and returns it as a dictionary.

        Returns:
            dict: A dictionary containing the file structure, main dataset, ancillary datasets, and metadata.
        """
        file_structure = {}

        # Open the HDF5 file
        with h5py.File(self.file, "r") as h5_f:
            # Structure of the HDF5 file
            def get_structure(name, obj):
                if isinstance(obj, h5py.Group):
                    file_structure[name] = {"type": "Group", "items": list(obj.keys())}
                elif isinstance(obj, h5py.Dataset):
                    file_structure[name] = {
                        "type": "Dataset",
                        "shape": obj.shape,
                        "dtype": str(obj.dtype),
                    }

            # Populate the file structure
            h5_f.visititems(get_structure)
            
            # Metadata (attributes) from /Measurement_000 group
            metadata = {}
            for key, value in h5_f["/Measurement_000"].attrs.items():
                # Convert possible numpy objects in attributes to native Python types
                if isinstance(value, np.ndarray):
                    metadata[key] = value.tolist()
                else:
                    metadata[key] = value

            file_structure["metadata"] = metadata

        return file_structure

    def get_tree(self):
        """
        get_tree reads the tree from the H5 file

        Returns:
            list: list of the tree from the H5 file
        """

        with h5py.File(self.file, "r+") as h5_f:
            return get_tree(h5_f)

    @static_state_decorator
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

    ##### GETTERS #####

    @property
    def get_state(self):
        """
        get_state function that return the dictionary of the current state

        Returns:
            dict: dictionary of the current state
        """
        return {
            "raw_format": self.raw_format,
            "fitter": self.fitter,
            "scaled": self.scaled,
            "output_shape": self.output_shape,
            "measurement_state": self.measurement_state,
            "LSQF_phase_shift": self.LSQF_phase_shift,
            "NN_phase_shift": self.NN_phase_shift,
            "noise": self.noise,
            "loop_interpolated": self.loop_interpolated,
        }

    @property
    def num_pix(self):
        """Number of pixels in the data"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["num_pix"]

    @property
    def num_pix_1d(self):
        """Number of pixels in the data"""
        with h5py.File(self.file, "r+") as h5_f:
            return int(np.sqrt(self.num_pix))

    @property
    def num_bins(self):
        """Number of frequency bins in the data"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["num_bins"]

    @property
    def spectroscopic_length(self):
        """Gets the length of the spectroscopic vector"""
        return self.num_bins * self.voltage_steps

    @property
    def be_repeats(self):
        """Number of BE repeats"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["BE_repeats"]

    @property
    def sampling_rate(self):
        """Sampling rate in Hz"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["IO_rate_[Hz]"]

    @property
    def be_center_frequency(self):
        """BE center frequency in Hz"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["BE_center_frequency_[Hz]"]

    @property
    def be_bandwidth(self):
        """BE bandwidth in Hz"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["BE_band_width_[Hz]"]

    @property
    def dc_voltage(self):
        """Gets the DC voltage vector"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f[f"Raw_Data_SHO_Fit/Raw_Data-SHO_Fit_000/Spectroscopic_Values"][
                0, 1::2
            ]

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
            # Retrieve the number of cycles from the attributes of "Measurement_000"
            cycles = h5_f["Measurement_000"].attrs["VS_number_of_cycles"]

            # Check if the measurement was performed 'in and out-of-field'
            # If so, double the number of cycles to account for both directions
            if (
                h5_f["Measurement_000"].attrs["VS_measure_in_field_loops"]
                == "in and out-of-field"
            ):
                cycles *= 2

            # Return the total number of cycles
            return cycles

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

    @property
    def frequency_bin(self):
        """Frequency bin vector in Hz"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"]["Channel_000"]["Bin_Frequencies"][:]

    @property
    def spectroscopic_values(self):
        """Spectroscopic values"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"]["Channel_000"]["Spectroscopic_Values"][:]

    @property
    def voltage_steps_per_cycle(self):
        """
        Retrieves the number of voltage steps per cycle.

        This function determines the number of voltage steps per cycle.

        Returns:
            int: The number of voltage steps per cycle corresponding to the current measurement state.
        """
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["VS_steps_per_full_cycle"]

    @property
    def be_waveform(self):
        """BE excitation waveform"""
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f["Measurement_000"]["Channel_000"]["Excitation_Waveform"][:]

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
            
    def get_loop_path(self):
        """
        get_loop_path gets the path where the hysteresis loops are located

        Returns:
            str: string pointing to the path where the hysteresis loops are located
        """

        if self.noise == 0 or self.noise is None:
            prefix = 'Raw_Data'
            return f"Measurement_000/{prefix}-SHO_Fit_000/Fit-Loop_Fit_000"
        else:
            prefix = f"Noisy_Data_{self.noise}"
            return f"/Noisy_Data_{self.noise}_SHO_Fit/Noisy_Data_{self.noise}-SHO_Fit_000/Guess-Loop_Fit_000"

    def raw_data_resampled(self, pixel=None, voltage_step=None):
        """
        raw_data_resampled Resampled real part of the complex data resampled

        Args:
            pixel (int, optional): selected pixel of data to resample. Defaults to None.
            voltage_step (int, optional): selected voltage step of data to resample. Defaults to None.

        Returns:
            np.array: resampled data
        """

        if pixel is not None and voltage_step is not None:
            return self.resampled_data[self.dataset][[pixel], :, :][
                :, [voltage_step], :
            ]
        else:
            with h5py.File(self.file, "r+") as h5_f:
                return self.resampled_data[self.dataset][:]

    def state_num_voltage_steps(self):
        """
        Retrieves the number of voltage steps based on the current measurement state.

        This function determines the number of voltage steps to use depending on whether
        the current measurement state is set to 'all' or a subset. If the measurement state
        is 'all', it returns the total number of voltage steps; otherwise, it returns half
        the total number.

        Returns:
            int: The number of voltage steps corresponding to the current measurement state.
        """

        # Check if the current measurement state is set to 'all'
        if self.measurement_state == "all":
            # If 'all', return the full number of voltage steps
            voltage_step = self.voltage_steps
        else:
            # If not 'all', return half the number of voltage steps
            voltage_step = int(self.voltage_steps / 2)

        # Return the computed number of voltage steps
        return voltage_step

    def get_freq_values(self, data):
        """
        get_freq_values Function that gets the frequency bins

        Args:
            data (np.array): BE data

        Raises:
            ValueError: original data and frequency bin mismatch

        Returns:
            np.array: frequency bins for the data
        """

        try:
            data = data.flatten()
        except:
            pass

        if np.isscalar(data) or len(data) == 1:
            length = data
        else:
            length = len(data)

        # checks if the length of the data is the raw length, or the resampled length
        if length == self.num_bins:
            x = self.frequency_bin
        elif length == self.resampled_bins:
            x = resample(self.frequency_bin, self.resampled_bins)
        else:
            raise ValueError(
                "original data must be the same length as the frequency bins or the resampled frequency bins"
            )
        return x

    def get_cycle(self, data, axis=0, **kwargs):
        """
        Extracts data for a specific cycle from the hysteresis loop.

        Args:
            data (np.array): The band excitation data from which to extract the cycle.
                            This is typically a multi-dimensional array containing several cycles.
            axis (int, optional): The axis along which to split the data into cycles. Defaults to 0.
            **kwargs: Additional keyword arguments to pass to np.array_split for custom behavior.

        Returns:
            np.array: The data corresponding to the specific cycle set by `self.cycle`.

        This function splits the data into multiple cycles based on the attribute `self.num_cycles`
        and then returns the data for the specific cycle indicated by `self.cycle`.
        """

        # Split the input data along the specified axis into 'num_cycles' parts
        data = np.array_split(data, self.num_cycles, axis=axis, **kwargs)

        # Extract the data for the cycle specified by 'self.cycle' (1-based index)
        data = data[self.cycle - 1]

        # Return the data corresponding to the specified cycle
        return data

    def get_measurement_cycle(self, data, cycle=None, axis=1):
        """
        Retrieves the data for a specific measurement cycle from band excitation data.

        This function extracts a specific cycle from the provided band excitation data.
        If a cycle number is provided, it updates the current cycle. The data is first
        processed based on the voltage state, and then the corresponding cycle is extracted
        using the `get_cycle` method.

        Args:
            data (np.array): The band excitation data to extract the cycle from. Typically a multi-dimensional array.
            cycle (int, optional): The specific cycle to extract. If not provided, the default cycle stored in the object is used. Defaults to None.
            axis (int, optional): The axis where the cycle dimension is located. Defaults to 1.

        Returns:
            np.array: The data corresponding to the specified measurement cycle.

        Notes:
            - The cycle number can be updated dynamically if passed as an argument.
            - The function first processes the data through `get_data_w_voltage_state` to align it with the correct voltage state.
            - Finally, it extracts the cycle using the `get_cycle` method.
        """

        # If a specific cycle is provided, update the current cycle attribute
        if cycle is not None:
            self.cycle = cycle

        # Process the data to align with the voltage state
        data = self.get_data_w_voltage_state(data)

        # Extract and return the data corresponding to the specific cycle along the specified axis
        return self.get_cycle(data, axis=axis)

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

        # Open the HDF5 file in read+write mode
        with h5py.File(self.file, "r+") as h5_f:
            # Extract data based on provided pixel and voltage_step indices
            if pixel is not None and voltage_step is not None:
                # Specific pixel and voltage_step provided
                return self.raw_data_reshaped[self.dataset][[pixel], :, :][
                    :, [voltage_step], :
                ]
            else:
                # Return the entire dataset if pixel or voltage_step is not specified
                return self.raw_data_reshaped[self.dataset][:]

    @property
    def get_original_data(self):
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

    @static_state_decorator
    def raw_spectra(
        self,
        pixel=None,
        voltage_step=None,
        fit_results=None,
        frequency=False,
        noise=None,
        state=None,
    ):
        """
        Simplifies the retrieval of raw band excitation data.

        This method retrieves the raw spectral data for a given pixel and voltage step,
        with options for using resampled data, fit results, and frequency bins. It also
        allows the setting of noise levels and the extraction state.

        Args:
            pixel (int, optional):
                The pixel value to retrieve data for. If None, the data for all pixels is considered.
            voltage_step (int, optional):
                The voltage step to retrieve data for. If None, a step is chosen based on the dataset state.
            fit_results (np.array, optional):
                Provided fit results used to generate the raw spectra. If None, raw data is used directly.
            frequency (bool, optional):
                Whether to return the frequency bins along with the data. Defaults to False.
            noise (int, optional):
                Noise level to use in data extraction. If None, no noise adjustment is made.
            state (dict, optional):
                A dictionary defining the extraction state. If provided, attributes are set accordingly.

        Returns:
            np.array:
                The band excitation data. If `frequency=True`, returns a tuple of the data and frequency bins.
        """

        # Set the noise level if provided
        if noise is not None:
            self.noise = noise

        # Set the extraction state attributes if provided
        if state is not None:
            self.set_attributes(**state)

        # Open the HDF5 file for reading and writing
        with h5py.File(self.file, "r+") as h5_f:
            # Flag to determine if data reshaping is needed
            shaper_ = True

            # Determine the voltage step considering the current measurement state
            voltage_step = self.measurement_state_voltage(voltage_step)

            # Determine the number of bins and frequency values based on resampling status
            if self.resampled:
                bins = self.resampled_bins
                frequency_bins = self.get_freq_values(bins)
            else:
                bins = self.num_bins
                frequency_bins = self.get_freq_values(bins)

            # Retrieve the raw data based on whether fit results are provided
            if fit_results is None:
                if self.resampled:
                    data = self.raw_data_resampled(
                        pixel=pixel, voltage_step=voltage_step
                    )
                else:
                    data = self.raw_data(pixel=pixel, voltage_step=voltage_step)
            else:
                # Process the fit results to obtain raw spectra
                params_shape = fit_results.shape

                if isinstance(fit_results, np.ndarray):
                    fit_results = torch.from_numpy(fit_results)

                # Reshape the fit results for fitting functions
                params = fit_results.reshape(-1, 4)

                # TODO: DELETE IF WORKS
                # # Evaluate the fitting function to generate data
                # data = eval(
                #     f"self.SHO_fit_func_{self.fitter}(params, frequency_bins)"
                # )

                data = SHO_nn(params, frequency_bins)

                # Check if the full dataset was used and determine if reshaping is needed
                if bins * self.num_pix * self.voltage_steps * 2 == len(data.flatten()):
                    pass
                else:
                    shaper_ = False

                if shaper_:
                    data = self.shaper(data, pixel, voltage_step)

            # Further processing based on pixel and voltage_step conditions
            if shaper_:
                if pixel is None or voltage_step is None:
                    data = self.get_data_w_voltage_state(data)

            # Handle different raw data formats (complex, magnitude spectrum)
            if self.raw_format == "complex":
                # Apply scaling if enabled
                if self.scaled:
                    data = self.raw_data_scaler.transform(data.reshape(-1, bins))

                if shaper_:
                    data = self.shaper(data, pixel, voltage_step)

                # Separate real and imaginary components
                data = [np.real(data), np.imag(data)]

            elif self.raw_format == "magnitude spectrum":
                if shaper_:
                    data = self.shaper(data, pixel, voltage_step)

                # Calculate magnitude and phase
                data = [np.abs(data), np.angle(data)]

            # Convert tensors to numpy arrays if necessary
            try:
                data[0] = data[0].numpy()
                data[1] = data[1].numpy()
            except:
                pass

            # Return the data and optionally the frequency bins
            if frequency:
                return data, frequency_bins
            else:
                return data

        # TODO Modification
        # # Check if both pixel and voltage_step are provided
        # if pixel is not None and voltage_step is not None:
        #     # Open the HDF5 file in read+write mode
        #     with h5py.File(self.file, "r+") as h5_f:
        #         # Extract and return data for the specific pixel and voltage step
        #         return self.raw_data_reshaped[self.dataset][[pixel], :, :][:, [voltage_step], :]
        # else:
        #     # Open the HDF5 file in read+write mode
        #     with h5py.File(self.file, "r+") as h5_f:
        #         # Extract and return the entire dataset
        #         return self.raw_data_reshaped[self.dataset][:]

    def SHO_LSQF(self, pixel=None, voltage_step=None):
        """
        Retrieves the Simple Harmonic Oscillator (SHO) fit results using the Least Squares Fitting (LSQF) method.

        This function extracts the SHO fit results from the dataset stored in an HDF5 file. The results can be
        retrieved for a specific pixel and voltage step, or for the entire dataset, depending on the provided arguments.

        Args:
            pixel (int, optional): The index of the pixel for which the SHO fit results are to be extracted.
                                If None, results for all pixels will be returned. Defaults to None.
            voltage_step (int, optional): The index of the voltage step for which the SHO fit results are to be extracted.
                                        If None, results for all voltage steps will be returned. Defaults to None.

        Returns:
            np.array: The extracted SHO LSQF results. The shape of the returned array depends on the
                    combination of the pixel and voltage_step parameters.
        """

        # Open the HDF5 file containing the SHO LSQF data
        with h5py.File(self.file, "r+") as h5_f:
            # Copy the SHO LSQF data for the specific dataset
            dataset_ = self.SHO_LSQF_data[f"{self.dataset}-SHO_Fit_000"].copy()

            # If both pixel and voltage_step are provided, return the data for the specific pixel and voltage step
            if pixel is not None and voltage_step is not None:
                return self.get_data_w_voltage_state(dataset_[[pixel], :, :])[
                    :, [voltage_step], :
                ]

            # If only pixel is provided, return the data for the specific pixel across all voltage steps
            elif pixel is not None:
                return self.get_data_w_voltage_state(dataset_[[pixel], :, :])

            # If neither pixel nor voltage_step are provided, return the entire dataset
            else:
                return self.get_data_w_voltage_state(dataset_[:])

    def get_data_w_voltage_state(self, data):
        """
        get_data_w_voltage_state function to extract data given a voltage state either the on or off state

        Args:
            data (np.array): BE data

        Returns:
            np.array: BE data considering the voltage state
        """

        # only does this if getting the full dataset, will reduce to off and on state
        if self.measurement_state == "all":
            data = data
        elif self.measurement_state == "on":
            data = data[:, 1::2, :]
        elif self.measurement_state == "off":
            data = data[:, ::2, :]

        return data

    @static_state_decorator
    def SHO_fit_results(self, state=None, model=None, phase_shift=None, X_data=None):
        """
        Retrieves the SHO (Simple Harmonic Oscillator) fit results from the dataset, either
        by using a specified neural network model or a least squares fitting method.

        Args:
            state (dict, optional): A dictionary representing a specific measurement state.
                                    If provided, the dataset will be adjusted to this state before fitting.
                                    Defaults to None.
            model (nn.Module, optional): A neural network model to predict the SHO fit results.
                                        If not provided, a least squares fitting method is used.
                                        Defaults to None.
            phase_shift (float, optional): A value to shift the phase of the resulting data.
                                        If None, the default phase shift from the dataset's configuration is used.
                                        Defaults to None.
            X_data (np.array, optional): The frequency bins used for model prediction.
                                        If None and a model is provided, it will be generated from the dataset.
                                        Defaults to None.

        Returns:
            np.array: The SHO fit parameters, either in the shape of (index, SHO_params) or
                    (num_pix, num_voltage_steps, SHO_params), depending on the dataset configuration.
        """

        # Note: Removed pixel and voltage step indexing here

        # If a neural network model is not provided, use the Least Squares Fitting (LSQF) method
        if model is None:
            # Open the HDF5 file for reading the SHO fitting data
            with h5py.File(self.file, "r+") as h5_f:
                # If a state is provided, set the dataset attributes accordingly
                if state is not None:
                    self.set_attributes(**state)

                # Evaluate and retrieve the fitting data using the specified fitter (e.g., LSQF)
                data = eval(f"self.SHO_{self.fitter}()")

                # Store the original shape of the data for reshaping later
                data_shape = data.shape

                # Reshape the data to a 2D array with 4 columns (assumed to be the SHO parameters)
                data = data.reshape(-1, 4)

                # If a phase shift is specified in the dataset's fitter configuration and no
                # external phase shift is provided, apply the default phase shift
                if (
                    eval(f"self.{self.fitter}_phase_shift") is not None
                    and phase_shift is None
                ):
                    data[:, 3] = eval(
                        f"self.shift_phase(data[:, 3], self.{self.fitter}_phase_shift)"
                    )

                # Reshape the data back to its original shape
                data = data.reshape(data_shape)

                # If the dataset is scaled, apply the scaling transformation to the data
                if self.scaled:
                    data = self.SHO_scaler.transform(data.reshape(-1, 4)).reshape(
                        data_shape
                    )

        else:
            # If a model is provided, use it to predict the SHO parameters

            # If X_data is not provided, generate the necessary input data (X_data, Y_data) from the dataset
            if X_data is None:
                X_data, Y_data = self.NN_data()

            # Predict the SHO parameters using the model
            pred_data, scaled_param, data = model.predict(X_data)

            # If the dataset is scaled, use the scaled parameters as the final data
            if self.scaled:
                data = scaled_param

        # Apply an external phase shift if provided
        if phase_shift is not None:
            data[:, 3] = self.shift_phase(data[:, 3], phase_shift)

        # Return the data reshaped according to the output configuration
        if self.output_shape == "index":
            # Return data as a 2D array (index, SHO_params)
            return data.reshape(-1, 4)
        else:
            # Return data as a 3D array (num_pix, num_voltage_steps, SHO_params)
            return data.reshape(self.num_pix, self.state_num_voltage_steps(), 4)

    @static_state_decorator
    def get_raw_data_from_LSQF_SHO(self, model, index=None):
        """
        Extracts raw data from LSQF (Least Squares Fit) SHO (Simple Harmonic Oscillator) fits.

        Args:
            model (dict): A dictionary that defines the state for extracting the SHO fit results.
                        The dictionary typically contains information about the fitter and its parameters.
            index (int, optional): The index of the specific data point to extract. Defaults to None,
                                meaning all data will be returned.

        Returns:
            tuple:
                pred_data (numpy.ndarray): The predicted raw spectra data reconstructed from the SHO fits.
                params (numpy.ndarray): The SHO parameters used for reconstruction.

        This method extracts the unscaled SHO fit parameters, reconstructs the raw spectra, and optionally
        returns the result for a specific index.
        """

        # Set object attributes based on the provided model dictionary.
        self.set_attributes(**model)

        # Disable scaling of parameters for accurate reconstructions.
        self.scaled = False

        # Get the unscaled SHO fit results (shifted parameters).
        params_shifted = self.SHO_fit_results()

        # Set the phase shift of the current fitter to 0.
        # This ensures that the reconstructed results are not phase-shifted.
        exec(f"self.{model['fitter']}_phase_shift=0")

        # Retrieve the SHO fit parameters with zero phase shift.
        params = self.SHO_fit_results()

        # Re-enable scaling of parameters after fetching the fit results.
        self.scaled = True

        # Reconstruct the raw spectra based on the SHO fit parameters (scaled values).
        pred_data = self.raw_spectra(fit_results=params)

        # Construct an array containing amplitude and phase, formatted as [amplitude, phase].
        pred_data = np.array([pred_data[0], pred_data[1]])

        # Reshape the data to match the expected format of the package.
        pred_data = np.swapaxes(pred_data, 0, 1)  # Swap axis 0 and 1.
        pred_data = np.swapaxes(pred_data, 1, 2)  # Swap axis 1 and 2.

        # If an index is provided, extract only the specified data point and corresponding parameters.
        if index is not None:
            pred_data = pred_data[[index]]  # Select the data at the given index.
            params = params_shifted[
                [index]
            ]  # Select the shifted parameters at the given index.

        # Return the reconstructed spectra (pred_data) and the SHO parameters.
        return pred_data, params

    @property
    def extraction_state(self):
        """
        Prints the current extraction state of the dataset.

        This property method outputs a summary of the current settings and parameters
        related to the extraction state of the dataset. It includes information such
        as whether the data is resampled, the format of the raw data, the fitting method
        used, and various other state-related attributes.

        Args:
            None

        Returns:
            None
        """

        if self.verbose:
            # Print a formatted string that summarizes the current extraction state of the dataset
            print(
                f"""
            Dataset = {self.dataset}
            Resample = {self.resampled}
            Raw Format = {self.raw_format}
            Fitter = {self.fitter}
            Scaled = {self.scaled}
            Output Shape = {self.output_shape}
            Measurement State = {self.measurement_state}
            Resample Resampled = {self.resampled}
            Resample Bins = {self.resampled_bins}
            LSQF Phase Shift = {self.LSQF_phase_shift}
            NN Phase Shift = {self.NN_phase_shift}
            Noise Level = {self.noise}
            Loop Interpolated = {self.loop_interpolated}
            """
            )

    ##### SETTERS #####

    def set_attributes(self, **kwargs):
        """
        Sets multiple attributes of the object using key-value pairs provided as keyword arguments.

        This method allows for dynamic setting of object attributes based on the provided
        dictionary of keyword arguments (`kwargs`). Each key in the dictionary corresponds
        to an attribute name, and the associated value is assigned to that attribute.

        If the keyword 'noise' is present in the arguments, it will trigger the setter
        method for the 'noise' attribute, allowing for any associated logic to be executed.

        Args:
            **kwargs: Arbitrary keyword arguments where keys are the attribute names and
                    values are the attribute values to be set.

        Example:
            obj.set_attributes(attr1=value1, attr2=value2, noise=some_noise_value)
            This will set `obj.attr1` to `value1`, `obj.attr2` to `value2`, and `obj.noise`
            to `some_noise_value` (while invoking any custom logic in the `noise` setter).
        """

        # Iterate over each key-value pair in kwargs and set the corresponding attribute
        for key, value in kwargs.items():
            setattr(self, key, value)

        # If 'noise' is present in kwargs, this explicitly calls the setter for 'noise'
        if "noise" in kwargs:
            self.noise = kwargs["noise"]

    ##### Data Transformers ######

    @staticmethod
    def shift_phase(phase, shift_=None):
        """
        Shifts the phase of the dataset by a specified amount. This function adjusts the phase
        values to account for any phase shift, ensuring the phase values are wrapped correctly
        within the range of -π to π or π to 3π depending on the shift direction.

        Args:
            phase (np.array): Array of phase data to be shifted.
            shift_ (float, optional): The phase shift to apply, in radians. If None or 0,
                                    no shift is applied. Defaults to None.

        Returns:
            np.array: The phase-shifted data, with phase values adjusted according to the specified shift.
        """

        # If no shift is specified or the shift is 0, return the original phase data unchanged
        if shift_ is None or shift_ == 0:
            return phase
        else:
            shift = shift_

        # If the shift is positive, adjust the phase values
        if shift > 0:
            # Increment phase by π to handle phase wrapping
            phase_ = phase
            phase_ += np.pi

            # Adjust phase values greater than π by adding 2π
            phase_[phase_ <= shift] += 2 * np.pi

            # Subtract the shift and adjust by -π to wrap the phase back within the appropriate range
            phase__ = phase_ - shift - np.pi

        # If the shift is negative, adjust the phase values accordingly
        else:
            # Decrement phase by π to handle phase wrapping
            phase_ = phase
            phase_ -= np.pi

            # Adjust phase values less than -π by subtracting 2π
            phase_[phase_ >= shift] -= 2 * np.pi

            # Subtract the shift and adjust by +π to wrap the phase back within the appropriate range
            phase__ = phase_ - shift + np.pi

        return phase__

    def waveform_constructor(self):
        """
        Constructs a combined waveform by adding elements from a hysteresis waveform and
        a band excitation (BE) waveform.

        This method creates a new waveform by repeating and tiling the elements of the
        `hysteresis_waveform` and `be_waveform` arrays, respectively. Each element of
        the hysteresis waveform is combined with all elements of the BE waveform.

        Returns:
            np.array:
                The resulting combined waveform array.
        """

        # Repeat each element of 'hysteresis_waveform' for the length of 'be_waveform'
        hysteresis_waveform_repeated = np.repeat(
            self.hysteresis_waveform, len(self.be_waveform)
        )

        # Tile 'be_waveform' so that it repeats for each element in 'hysteresis_waveform'
        be_waveform_tiled = np.tile(self.be_waveform, len(self.hysteresis_waveform))

        # Combine the repeated and tiled arrays by adding them element-wise
        result = hysteresis_waveform_repeated + be_waveform_tiled

        # Return the resulting combined waveform
        return result

    def measurement_state_voltage(self, voltage_step):
        """
        Determines the voltage step index based on the measurement state.

        This function adjusts the provided voltage step index according to the current
        measurement state of the dataset (e.g., 'on' or 'off'). It returns the corresponding
        voltage step index based on the dataset's state.

        Args:
            voltage_step (int):
                The voltage step index to select.

        Returns:
            int:
                The adjusted voltage step index based on the measurement state.
        """

        if voltage_step is not None:
            # Adjust the voltage step index for the 'on' state by selecting odd-indexed steps
            if self.measurement_state == "on":
                voltage_step = np.arange(0, self.voltage_steps)[1::2][voltage_step]
            # Adjust the voltage step index for the 'off' state by selecting even-indexed steps
            elif self.measurement_state == "off":
                voltage_step = np.arange(0, self.voltage_steps)[::2][voltage_step]

        # Return the adjusted voltage step index
        return voltage_step

    def shaper(self, data, pixel=None, voltage_steps=None):
        """
        Reshapes band excitation (BE) data based on the current measurement state and specified parameters.

        This utility function reshapes the provided band excitation data according to the
        pixel and voltage step specifications, taking into account the current measurement
        state of the dataset. It handles different output shapes, including reshaping by
        pixels or by index.

        Args:
            data (np.array):
                The band excitation data to be reshaped. This is typically a multidimensional array.
            pixel (int or list of ints, optional):
                The pixel(s) to reshape the data for. If None, all pixels are considered.
                Defaults to None.
            voltage_steps (int or list of ints, optional):
                The voltage step(s) to reshape the data for. If None, all voltage steps
                are considered, adjusting for the measurement state. Defaults to None.

        Raises:
            ValueError:
                If an invalid output shape is provided. The output shape must be either 'pixels' or 'index'.

        Returns:
            np.array:
                The reshaped band excitation data.
        """

        # Determine the number of pixels to reshape for, handling cases where a single pixel or list of pixels is provided
        if pixel is not None:
            try:
                num_pix = len(pixel)  # If a list of pixels is provided, get the length
            except:
                num_pix = (
                    1  # If a single pixel is provided, set the number of pixels to 1
                )
        else:
            num_pix = int(
                self.num_pix.copy()
            )  # If no pixel is specified, use the total number of pixels

        # Determine the number of voltage steps to reshape for, handling cases where a single step or list of steps is provided
        if voltage_steps is not None:
            try:
                voltage_steps = len(
                    voltage_steps
                )  # If a list of voltage steps is provided, get the length
            except:
                voltage_steps = 1  # If a single voltage step is provided, set the number of voltage steps to 1
        else:
            voltage_steps = int(
                self.voltage_steps.copy()
            )  # If no voltage step is specified, use the total number of voltage steps

            # Adjust the number of voltage steps if the measurement state is "on" or "off"
            if self.measurement_state in ["on", "off"]:
                voltage_steps /= 2  # Halve the number of voltage steps if the measurement state is "on" or "off"
                voltage_steps = int(voltage_steps)

        # Reshape the data based on the specified output shape
        if self.output_shape == "pixels":
            data = data.reshape(num_pix, voltage_steps, -1)  # Reshape by pixels
        elif self.output_shape == "index":
            data = data.reshape(num_pix * voltage_steps, -1)  # Reshape by index
        else:
            raise ValueError(
                "output_shape must be either 'pixel' or 'index'"
            )  # Raise an error if an invalid output shape is specified

        return data

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

        # Open the HDF5 file for reading and writing
        with h5py.File(self.file, "r+") as h5_f:
            # Check if resampling is needed by comparing the number of bins
            if self.resampled_bins != self.num_bins:
                # Loop through each dataset to perform resampling
                for data in self.raw_datasets:
                    # Resample the data using the provided resampler function
                    resampled_ = self.resampler(
                        self.raw_data_reshaped[data].reshape(-1, self.num_bins), axis=2
                    )

                    # Reshape the resampled data to match the original dimensions
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

        # Open the HDF5 file for reading and writing
        with h5py.File(self.file, "r+") as h5_f:
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

    @staticmethod
    def to_real_imag(data):
        """
        Extracts the real and imaginary components from band excitation (BE) data.

        This function takes in BE data, which may be in either a NumPy array or a PyTorch
        tensor format, converts it to its complex form, and then separates the real and
        imaginary parts.

        Args:
            data (np.array or torch.Tensor): BE data, either as a NumPy array or a PyTorch tensor.

        Returns:
            list: A list containing two NumPy arrays: the first array represents the real
                components, and the second array represents the imaginary components
                of the BE response.
        """

        # Convert the data to its complex form using the to_complex method from the BE_Dataset class.
        data = BE_Dataset.to_complex(data)

        # Extract and return the real and imaginary components as a list of NumPy arrays.
        return [np.real(data), np.imag(data)]

    @staticmethod
    def to_magnitude(data):
        """
        Converts a complex number representing the photodiode response of the cantilever
        into its amplitude (magnitude) and phase.

        This function takes the complex photodiode response and calculates both the magnitude
        (amplitude) and phase using NumPy's `abs` and `angle` functions, respectively.

        Args:
            data (np.array): A NumPy array containing the complex photodiode response of the cantilever.

        Returns:
            list: A list of two NumPy arrays:
                - The first element is the magnitude of the cantilever response (amplitude).
                - The second element is the phase of the cantilever response (in radians).

        Example:
            If `data` contains complex numbers, this function will return:
            - `np.abs(data)` for the magnitude.
            - `np.angle(data)` for the phase.
        """

        # Convert the input data to its complex representation
        data = BE_Dataset.to_complex(data)

        # Calculate and return the magnitude (absolute value) and phase (angle) of the complex data
        return [np.abs(data), np.angle(data)]

    ##### NOISE GETTER and SETTER #####

    @property
    def noise(self):
        """Noise value"""
        return self._noise

    @noise.setter
    def noise(self, noise):
        """Sets the noise value"""
        self._noise = noise
        self.set_noise_state(noise)

    def set_noise_state(self, noise):
        """function that uses the noise state to set the current dataset

        Args:
            noise (int): noise value in multiples of the standard deviation

        Raises:
            ValueError: error if the noise value does not exist in the dataset
        """

        if noise == 0:
            self.dataset = "Raw_Data"
        else:
            self.dataset = f"Noisy_Data_{noise}"

    ##### Machine Learning Functions #####

    def test_train_split_(
        self, test_size=0.2, random_state=42, resampled=None, scaled=True, shuffle=True
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
        x_data, y_data = self.NN_data(resampled, scaled)

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

    @static_state_decorator
    def NN_data(self, resampled=None, scaled=True):
        """
        Utility function that retrieves and prepares the data for neural network training.

        Args:
            resampled (bool, optional): If True, use the resampled data; otherwise, use original data. Defaults to None.
            scaled (bool, optional): If True, use scaled data; otherwise, use unscaled data. Defaults to True.

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

        # Ensure the data is scaled if required, as scaling is often necessary for neural network training
        self.scaled = scaled

        # Retrieve the raw spectral data
        data = self.raw_spectra()

        # Convert the raw data into a format suitable for neural network input
        x_data = self.to_nn(data)

        # Retrieve the SHO fit results, which are scaled LSQF parameters
        y_data = self.SHO_fit_results().reshape(-1, 4)

        # Convert the LSQF results into a tensor for use in training and evaluation
        y_data = torch.tensor(y_data, dtype=torch.float32)

        # Return the neural network input data and corresponding fit parameters
        return x_data, y_data
    
    ##### Hysteresis Functions #####
    
    @static_state_decorator
    def get_hysteresis(self,
                       fits = False,
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
                h5_projected_loops = h5_f[ h5_path + '/Projected_Loops']
            else:
                h5_projected_loops = h5_f[ h5_path + '/Fit']

            # Prepare some variables for plotting loops fits and guesses
            # Plot the Loop Guess and Fit Results
            proj_nd, _ = reshape_to_n_dims(h5_projected_loops)

            spec_ind = get_auxiliary_datasets(h5_projected_loops,
                                              aux_dset_name='Spectroscopic_Indices')[-1]
            spec_values = get_auxiliary_datasets(h5_projected_loops,
                                                 aux_dset_name='Spectroscopic_Values')[-1]
            pos_ind = get_auxiliary_datasets(h5_projected_loops,
                                             aux_dset_name='Position_Indices')[-1]

            pos_nd, _ = reshape_to_n_dims(pos_ind, h5_pos=pos_ind)
            pos_dims = list(pos_nd.shape[:pos_ind.shape[1]])

            # reshape the vdc_vec into DC_step by Loop
            spec_nd, _ = reshape_to_n_dims(spec_values, h5_spec=spec_ind)
            loop_spec_dims = np.array(spec_nd.shape[1:])
            loop_spec_labels = sidpy.hdf.hdf_utils.get_attr(spec_values, 'labels')

            spec_step_dim_ind = np.where(loop_spec_labels == 'DC_Offset')[0][0]

            # Also reshape the projected loops to Positions-DC_Step-Loop
            final_loop_shape = pos_dims + \
                [loop_spec_dims[spec_step_dim_ind]] + [-1]
            proj_nd2 = np.moveaxis(
                proj_nd, spec_step_dim_ind + len(pos_dims), len(pos_dims))
            proj_nd_3 = np.reshape(proj_nd2, final_loop_shape)

            # Get the bias vector:
            spec_nd2 = np.moveaxis(
                spec_nd[spec_step_dim_ind], spec_step_dim_ind, 0)
            bias_vec = np.reshape(spec_nd2, final_loop_shape[len(pos_dims):])

            if plotting_values:
                proj_nd_3, bias_vec = self.roll_hysteresis(bias_vec, proj_nd_3)

            hysteresis_data = np.transpose(proj_nd_3, (1, 0, 3, 2))

            # interpolates the data
            if self.loop_interpolated:
                hysteresis_data = clean_interpolate(hysteresis_data)

            # transforms the data with the scaler if necessary.
            if self.scaled:
                hysteresis_data = self.hysteresis_scaler_.transform(
                    hysteresis_data)

            # sets the data to the correct output shape
            if self.output_shape == "index":
                hysteresis_data = proj_nd_3.reshape(
                    self.num_cycles*self.num_pix, self.voltage_steps//self.num_cycles)
            elif self.output_shape == "pixels":
                pass

            hysteresis_data = self.hysteresis_measurement_state(
                hysteresis_data)

        # output shape (x,y, cycle, voltage_steps)
        # bias_vec
        return hysteresis_data, np.swapaxes(np.atleast_2d(self.get_voltage), 0, 1).astype(np.float64)

    
    def get_LSQF_hysteresis_fits(self, compare=False, index=True):
        """
        Retrieves the least squares quadratic fit hysteresis loops.
        Args:
            compare (bool, optional): If True, returns the fitted loops, raw hysteresis loops, and voltage values.
                                     If False, returns only the fitted loops. Defaults to False.
        Returns:
            numpy.ndarray or tuple: If compare is True, returns a tuple containing the fitted loops,
                                    raw hysteresis loops, and voltage values.
                                    If compare is False, returns only the fitted loops.
        """
        raw_hysteresis_loops, voltage = self.get_hysteresis(scaled=True, loop_interpolated = True)

        if index is True:
            raw_hysteresis_loops = raw_hysteresis_loops.reshape(-1,self.voltage_steps_per_cycle)

        # TODO: Update so not hard coded
        params = self.LSQF_hysteresis_params().reshape(-1, 9)

        loops = self.hysteresis_function(params, voltage[:,0].squeeze()).to(
                'cpu').detach().numpy().squeeze()

        if index is False:
            loops = loops.reshape(raw_hysteresis_loops.shape)

        if compare:
            return loops, raw_hysteresis_loops, voltage

        return loops

    def loop_fit_preprocessing(self):
        """
        loop_fit_preprocessing preprocessing for the loop fit results
        """

        # gets the hysteresis loops
        hysteresis, bias = self.get_hysteresis(
            plotting_values=True, output_shape="index")

        # interpolates any missing points in the data
        cleaned_hysteresis = clean_interpolate(hysteresis)

        # instantiates and computes the global scaler
        self.hysteresis_scaler_ = GlobalScaler()
        self.hysteresis_scaler_.fit_transform(cleaned_hysteresis)

        try:
            self.LoopParmScaler()
        except:
            pass

    @property
    def hysteresis_scaler(self):
        """
        get_hysteresis_scaler gets the hysteresis scaler

        Returns:
            scaler: scaler for the hysteresis loops
        """

        return self.hysteresis_scaler_

    @property
    def get_voltage(self):
        """
        get_voltage gets the voltage vector

        Returns:
            np.array: voltage vector
        """
        
        # TODO: Look for a way to refactor and not hard code. 
        with h5py.File(self.file, "r+") as h5_f:
            return h5_f['Measurement_000']['Channel_000']['UDVS'][::2][:, 1][24:120] * -1

    # @property
    # def get_hysteresis_voltage_len(self):
    #     """
    #     Get the length of the voltage vector for hysteresis measurements.

    #     Returns:
    #         int: Length of the voltage vector.
    #     """
    #     return self.get_voltage.shape[0]  # Return the length of the voltage vector

    # def default_state(self):
    #     """
    #     default_state Function that returns the dataset to the default state
    #     """

    #     # dictionary of the default state
    #     default_state_ = {'raw_format': "complex",
    #                       "fitter": 'LSQF',
    #                       "output_shape": "pixels",
    #                       "scaled": False,
    #                       "measurement_state": "all",
    #                       "resampled": False,
    #                       "resampled_bins": 80,
    #                       "LSQF_phase_shift": None,
    #                       "NN_phase_shift": None, }

    #     # sets the attributes to the default state
    #     self.set_attributes(**default_state_)

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

    # # delete a dataset
    # def delete(self, name):
    #     """
    #     delete function to delete a dataset within a pyUSID file

    #     Args:
    #         name (str): path of dataset to delete
    #     """

    #     with h5py.File(self.file, "r+") as h5_f:
    #         try:
    #             del h5_f[name]
    #         except KeyError:
    #             print("Dataset not found, could not be deleted")

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

    def LSQF_Loop_Fit(self,
                      main_dataset=None,
                      h5_target_group=None,
                      max_cores=None,
                      force=False,
                      h5_sho_targ_grp=None):
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
            if main_dataset is None:
                h5_main = usid.hdf_utils.find_dataset(
                    h5_file, 'Raw_Data')[0]
            else:
                h5_main = usid.hdf_utils.find_dataset(
                    h5_file, main_dataset)[0]

            # gets the measurement group name
            h5_meas_grp = h5_main.parent.parent

            # does the SHO_fit if it does not exist.
            sho_fit_points = 5  # The number of data points at each step to use when fitting
            sho_override = False  # Force recompute if True
            sho_fitter = belib.analysis.BESHOfitter(
                h5_main, cores=max_cores, verbose=False, h5_target_group=h5_meas_grp)
            sho_fitter.set_up_guess(
                guess_func=belib.analysis.be_sho_fitter.SHOGuessFunc.complex_gaussian, num_points=sho_fit_points)
            h5_sho_guess = sho_fitter.do_guess(override=sho_override)
            sho_fitter.set_up_fit()
            h5_sho_fit = sho_fitter.do_fit(override=sho_override)
            h5_sho_grp = h5_sho_fit.parent

            # gets the experiment type from the file
            expt_type = sidpy.hdf.hdf_utils.get_attr(h5_file, 'data_type')

            # finds the dataset from the file
            h5_meas_grp = usid.hdf_utils.find_dataset(
                h5_file, self.measure_group())

            # extract the voltage mode
            vs_mode = sidpy.hdf.hdf_utils.get_attr(
                h5_file["/Measurement_000"], 'VS_mode')

            try:
                vs_cycle_frac = sidpy.hdf.hdf_utils.get_attr(
                    h5_file["/Measurement_000"], 'VS_cycle_fraction')

            except KeyError:
                print('VS cycle fraction could not be found. Setting to default value')
                vs_cycle_frac = 'full'

            sho_fit, sho_dataset = self.SHO_Fitter(fit_group=True)

            # instantiates the loop fitter using belib
            loop_fitter = belib.analysis.BELoopFitter(h5_sho_fit,
                                                      expt_type, vs_mode, vs_cycle_frac,
                                                        #  h5_target_group=h5_meas_grp,
                                                      cores=max_cores,
                                                      verbose=False)

            # computes the guess for the loop fits
            loop_fitter.set_up_guess()
            h5_loop_guess = loop_fitter.do_guess(override=force)

            # Calling explicitly here since Fitter won't do it automatically
            h5_guess_loop_parms = loop_fitter.extract_loop_parameters(
                h5_loop_guess)
            loop_fitter.set_up_fit()
            h5_loop_fit = loop_fitter.do_fit(override=force)

            # save the path where the loop fit results are saved
            h5_loop_group = h5_loop_fit.parent

        return h5_loop_fit, h5_loop_group

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

    # @property
    # def resampled_freq(self):
    #     """Gets the resampled frequency"""
    #     return resample(self.frequency_bin, self.resampled_bins)

    @static_state_decorator
    def LSQF_hysteresis_params(self, output_shape=None, scaled=None, measurement_state=None):
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
            data = h5_f[f"/Measurement_000/{self.dataset}-SHO_Fit_000/Fit-Loop_Fit_000/Fit"][:]
            data = data.reshape(self.num_rows, self.num_cols, self.num_cycles)
            data = np.array([data['a_0'], data['a_1'], data['a_2'], data['a_3'], data['a_4'],
                            data['b_0'], data['b_1'], data['b_2'], data['b_3']]).transpose((1, 2, 3, 0))

            if self.scaled:
                # TODO: add the scaling here
                data = self.loop_param_scaler.fit(data)

                # Warning("Scaling not implemented yet")
                # pass

            if self.output_shape == "index":
                data = data.reshape(
                    self.num_pix, self.num_cycles, data.shape[-1])

            data = self.hysteresis_measurement_state(data)

            return data

    def LoopParmScaler(self):
        
        # TODO: Could update not to be hard coded
        self.loop_param_scaler = StandardScaler()
        data = self.LSQF_hysteresis_params().reshape(-1, 9)

        self.loop_param_scaler.fit(data)



    
    # def get_bias_vector(self, plotting_values=True):

    #     # TODO: could look at get_hysteresis to simplify code

    #     with h5py.File(self.file, "r+") as h5_f:

    #         # gets the path where the hysteresis loops are located
    #         h5_path = self.get_loop_path()

    #         # gets the projected loops
    #         h5_projected_loops = h5_f[h5_path + '/Projected_Loops']

    #         spec_ind = get_auxiliary_datasets(h5_projected_loops,
    #                                           aux_dset_name='Spectroscopic_Indices')[-1]
    #         pos_ind = get_auxiliary_datasets(h5_projected_loops,
    #                                          aux_dset_name='Position_Indices')[-1]
    #         spec_values = get_auxiliary_datasets(h5_projected_loops,
    #                                              aux_dset_name='Spectroscopic_Values')[-1]

    #         pos_nd, _ = reshape_to_n_dims(pos_ind, h5_pos=pos_ind)
    #         pos_dims = list(pos_nd.shape[:pos_ind.shape[1]])

    #         pos_dims = list(pos_nd.shape[:pos_ind.shape[1]])

    #         # reshape the vdc_vec into DC_step by Loop
    #         spec_nd, _ = reshape_to_n_dims(spec_values, h5_spec=spec_ind)
    #         loop_spec_labels = get_attr(spec_values, 'labels')
    #         spec_step_dim_ind = np.where(loop_spec_labels == 'DC_Offset')[0][0]

    #         loop_spec_dims = np.array(spec_nd.shape[1:])

    #         # Also reshape the projected loops to Positions-DC_Step-Loop
    #         final_loop_shape = pos_dims + \
    #             [loop_spec_dims[spec_step_dim_ind]] + [-1]

    #         # Get the bias vector:
    #         spec_nd2 = np.moveaxis(
    #             spec_nd[spec_step_dim_ind], spec_step_dim_ind, 0)

    #         bias_vec = np.reshape(spec_nd2, final_loop_shape[len(pos_dims):])

    #         if plotting_values:
    #             bias_vec = self.roll_hysteresis(bias_vec)

    #         return bias_vec

    def hysteresis_measurement_state(self, hysteresis_data):
        """utility function to extract the measurement state from the hysteresis data

        Args:
            hysteresis_data (np.array): hysteresis data to extract the measurement state from

        Returns:
            np.array: hysteresis data with the measurement state extracted
        """

        if self.measurement_state == "all" or self.measurement_state is None:
            return hysteresis_data
        if self.measurement_state == "off":
            return hysteresis_data[:, :, hysteresis_data.shape[2]//2:hysteresis_data.shape[2], :]
        if self.measurement_state == "on":
            return hysteresis_data[:, :, 0:hysteresis_data.shape[2]//2, :]

    def roll_hysteresis(self, bias_vector, hysteresis=None,
                        shift=4):
        """
        roll_hysteresis function to shift the bias vector and the hysteresis loop by a quarter cycle. This is to compensate for the difference in how the data is stored.

        Args:
            hysteresis (np.array): array for the hysteresis loop
            bias_vector (np.array): array for the bias vector
            shift (int, optional): fraction to roll the hysteresis loop by. Defaults to 4.

        Returns:
            _type_: _description_
        """

        # TODO: long term this is likely the wrong way to do this, should get this from the USID file spectroscopic index

        # Shift the bias vector and the loops by a quarter cycle
        shift_ind = int(-1 * bias_vector.shape[0] / shift)
        bias_vector = np.roll(bias_vector, shift_ind, axis=0)
        if hysteresis is None:
            return bias_vector
        else:
            proj_nd_shifted = np.roll(hysteresis, shift_ind, axis=2)
            return proj_nd_shifted, bias_vector

    # @property
    # def BE_superposition_state(self):
    #     """
    #     BE_superposition_state get the BE superposition state

    #     Returns:
    #         str: gets the superposition state
    #     """
    #     with h5py.File(self.file, "r+") as h5_f:
    #         BE_superposition_state_ = h5_f["Measurement_000"].attrs['VS_measure_in_field_loops']
    #     return BE_superposition_state_

    # def loop_shaper(self, data, shape="pixels"):
    #     """
    #     loop_shaper Tool to reshape the piezoelectric hysteresis loops based on the desired shape

    #     Args:
    #         data (np.array): hysteresis loops to reshape
    #         shape (str, optional): pixel or index as a string to reshape. Defaults to "pixels".

    #     Raises:
    #         ValueError: The data shape is not compatible with the number of rows and columns
    #         ValueError: The data shape is not compatible with the number of rows and columns

    #     Returns:
    #         np.array: reshaped piezoelectric hysteresis loops.
    #     """

    #     if shape == "pixels":
    #         try:
    #             return data.reshape(self.rows, self.cols, self.voltage_steps, self.num_cycles)
    #         except:
    #             raise ValueError(
    #                 "The data shape is not compatible with the number of rows and columns")
    #     if shape == "index":
    #         try:
    #             return data.reshape(self.num_pix_1d, self.voltage_steps, self.num_cycles)
    #         except:
    #             raise ValueError(
    #                 "The data shape is not compatible with the number of rows and columns")

    

    # def hysteresis_tensor(self, data):
    #     """
    #     hysteresis_tensor utility function that converts data to a tensor

    #     Args:
    #         data (np.array): data to convert to a tensor

    #     Returns:
    #         torch.tensor: tensor of the data
    #     """
    #     return torch.atleast_3d(torch.tensor(data.reshape(-1, self.get_hysteresis_voltage_len)))

    # def print_hysteresis_mse(self, model, data, labels):

    #     data = tuple(self.hysteresis_tensor(item) for item in data)

    #     model.print_mse(data, labels, is_SHO=False)
