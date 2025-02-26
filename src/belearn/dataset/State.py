from typing import Optional, Dict, Any
from dataclasses import field
from belearn.util.wrappers import static_state_decorator
import h5py
import numpy as np
import torch
from belearn.functions.sho import SHO_nn
from belearn.dataset.dataset_new import BE_Dataset
from belearn.dataset.preprocessing import Preprocessing

from scipy.signal import resample

class State(Preprocessing):
    # None of these are actually used in the class, but they are here to be (hopefully)used in the future
    
    
    def __init__(self,
                noise: int = 0,
                raw_format: str = "complex",
                fitter: str = "LSQF",
                scaled: bool = False,
                output_shape: str = "pixels",
                measurement_state: str = "all",
                loop_interpolated: bool = False,
                LSQF_phase_shift: Optional[float] = None,
                NN_phase_shift: Optional[float] = None,
                verbose: bool = False,
                resampled: bool = False,
                resampled_bins: Optional[int] = field(default=None, init=False),
                resampled_data: Dict[str, Any] = field(default_factory=dict, init=False)
            ):
        super().__init__(noise=noise)
        self.noise = noise
        self.raw_format = raw_format
        self.fitter = fitter
        self.scaled = scaled
        self.output_shape = output_shape
        self.measurement_state = measurement_state
        self.loop_interpolated = loop_interpolated
        self.LSQF_phase_shift = LSQF_phase_shift
        self.NN_phase_shift = NN_phase_shift
        self.verbose = verbose
        self.resampled = resampled
        self.resampled_bins = resampled_bins
        self.resampled_data = resampled_data
        
        self.set_raw_data()
    
    
    

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
        #with h5py.File(self.file, "r+") as h5_f:
        # consequently I also unindented the relevant code
        # Extract data based on provided pixel and voltage_step indices
        if pixel is not None and voltage_step is not None:
            # Specific pixel and voltage_step provided
            return self.raw_data_reshaped[self.dataset_name][[pixel], :, :][
                :, [voltage_step], :
            ]
        else:
            # Return the entire dataset if pixel or voltage_step is not specified
            return self.raw_data_reshaped[self.dataset_name][:]

    
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
            return self.resampled_data[self.dataset_name][[pixel], :, :][
                :, [voltage_step], :
            ]
        else:
            # JGoddy commented out the h5py file opening because
            # h5_f was not being used in the code
            #with h5py.File(self.file, "r+") as h5_f:
            return self.resampled_data[self.dataset_name][:]

   
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
        # JGoddy commented out the h5py file opening because
        # h5_f was not being used in the code
        # consequently I also unindented the relevant code
        #with h5py.File(self.file, "r+") as h5_f: 
        
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
            
            # JGoddy commented out the h5py file opening because
            # h5_f was not being used in the code
            #with h5py.File(self.file, "r+") as h5_f:
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
    
    ##### Decorators #####

    def static_dataset_decorator(func):
        """
        Decorator that preserves the dataset's state before and after a function call.

        This decorator ensures that the state of the dataset remains unchanged after
        the decorated function is executed. It captures the current state before the
        function is called and restores it afterward.

        Args:
            func (method):
                The method to be decorated. This can be any method that interacts with
                the dataset and might alter its state.

        Returns:
            method:
                The wrapped function that preserves the dataset's state.
        """

        def wrapper(*args, **kwargs):
            # Capture the current state of the dataset
            current_state = args[0].get_state

            # Execute the decorated function and capture its output
            out = func(*args, **kwargs)

            # Restore the dataset's state to what it was before the function was called
            args[0].set_attributes(**current_state)

            # Return the output of the function
            return out

        return wrapper
    
    def static_scale_decorator(func):
            """
            Decorator that preserves the state of the `SHO_ranges` and the dataset attributes
            before the decorated function is called and restores them afterward. This ensures
            that the function does not alter the state of the object it operates on.

            Args:
                func (method): The method to be decorated.

            Returns:
                method: The wrapped method with state-preservation functionality.
            """

            def wrapper(self, SHO_data, *args, **kwargs):
                """
                Wrapper function that preserves the current `SHO_ranges` and dataset state,
                calls the original function, and then restores the preserved state.

                Args:
                    self: Instance of the class containing the method.
                    SHO_data: Data to be processed by the wrapped function.
                    *args: Additional positional arguments passed to the wrapped function.
                    **kwargs: Additional keyword arguments passed to the wrapped function.

                Returns:
                    Any: The output of the wrapped function.
                """

                # Preserve the current SHO_ranges
                current_SHO_ranges = self.SHO_ranges

                # Preserve the current state of the dataset (assuming get_state returns a dictionary)
                current_dataset_state = (
                    self.get_state
                )  # Assume this returns a dict of the dataset state

                # Debugging output to verify the preserved state
                print("current_SHO_ranges:", current_SHO_ranges)
                print("current_dataset_state:", current_dataset_state)

                # Call the original function with the given arguments
                out = func(self, SHO_data, *args, **kwargs)

                # Restore the preserved SHO_ranges
                self.SHO_ranges = current_SHO_ranges

                # Restore the preserved dataset state by setting the attributes back to their original values
                self.set_attributes(**current_dataset_state)

                return out

            return wrapper
