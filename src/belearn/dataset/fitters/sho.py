import time
from belearn.dataset.fitters.base import BaseFitter
import h5py
import os
import pyUSID as usid
import sidpy
from BGlib import be as belib
import torch

class SHOFitter(BaseFitter):
    
    def fit(self,
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
    
    def fit_all(self, *args, **kwargs):
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
            
            
def SHO_fit_func_nn(params,
                wvec_freq,
                device='cpu'):
    """_summary_

    Returns:
        _type_: _description_
    """

    Amp = params[:, 0].type(torch.complex128)
    w_0 = params[:, 1].type(torch.complex128)
    Q = params[:, 2].type(torch.complex128)
    phi = params[:, 3].type(torch.complex128)
    wvec_freq = torch.tensor(wvec_freq)

    Amp = torch.unsqueeze(Amp, 1)
    w_0 = torch.unsqueeze(w_0, 1)
    phi = torch.unsqueeze(phi, 1)
    Q = torch.unsqueeze(Q, 1)

    wvec_freq = wvec_freq.to(device)

    numer = Amp * torch.exp((1.j) * phi) * torch.square(w_0)
    den_1 = torch.square(wvec_freq)
    den_2 = (1.j) * wvec_freq.to(device) * w_0 / Q
    den_3 = torch.square(w_0)

    den = den_1 - den_2 - den_3

    func = numer / den

    return func