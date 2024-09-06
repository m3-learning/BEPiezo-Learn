# import numpy as np
# from m3_learning.nn.Fitter1D.Fitter1D import Model
# from m3_learning.viz.layout import (
#     layout_fig,
#     inset_connector,
#     add_box,
#     subfigures,
#     add_text_to_figure,
#     get_axis_pos_inches,
#     imagemap,
#     FigDimConverter,
#     labelfigs,
#     imagemap,
#     scalebar,
# )
# from scipy.signal import resample
# from scipy import fftpack

# from m3_learning.be.nn import SHO_Model
# from m3_learning.be.loop_fitter import loop_fitting_function_torch
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from m3_learning.nn.Fitter1D.Fitter1D import Model
# from m3_learning.nn.Fitter1D import Fitter1D
# # import m3_learning
# # from m3_learning.util.rand_util import get_tuple_names
# import torch
# from torch import nn
# import pandas as pd
# import seaborn as sns
# from m3_learning.util.file_IO import make_folder
# from m3_learning.viz.Movies import make_movie
# import os

import os
import torch

from m3util.viz.layout import layout_fig, add_box, inset_connector, add_text_to_figure, labelfigs, scalebar, imagemap, FigDimConverter, subfigures
from m3util.util.IO import make_folder
from m3util.viz.movies import make_movie

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Type
import numpy as np
from scipy import fftpack
from torch import nn
from belearn.dataset.analytics import get_rankings



import matplotlib.pyplot as plt


# Defines the color palets for the plots
color_palette = {
    "LSQF_A": "#003f5c",
    "LSQF_P": "#444e86",
    "NN_A": "#955196",
    "NN_P": "#dd5182",
    "other": "#ff6e54",
    "other_2": "#ffa600",
}


def get_lowest_loss_for_noise_level(path, desired_noise_level):
    """
    Retrieves the filename of the checkpoint file with the lowest training loss for a specified noise level.

    This function searches through checkpoint files in the specified directory, extracts the noise level and loss value 
    from the filenames, and returns the filename with the lowest loss for the given noise level.

    Args:
        path (str): The directory path where the checkpoint files (.pth) are located.
        desired_noise_level (int or str): The noise level to search for. Can be provided as an integer or string.

    Returns:
        str: The filename with the lowest loss for the desired noise level. 
             Returns None if no files match the desired noise level.
    """

    # Checks if the desired noise level is provided as an integer
    if isinstance(desired_noise_level, int):
        # Converts the integer noise level to a string to match the filename format
        desired_noise_level = str(desired_noise_level)

    # Initialize a dictionary to store the lowest loss and corresponding filename for each noise level
    lowest_losses = {}

    # Iterate over all files in the directory
    for root, dirs, files in os.walk(path):

        # Loop through each file found in the directory
        for file in files:

            # Process only checkpoint files with the ".pth" extension
            if file.endswith(".pth"):

                # Extract the noise value from the filename
                noise_value = file.split("_noise_")[1].split("_")[0]

                # Extract the loss value from the filename and convert it to a float
                loss = file.split("train_loss_")[1].split("_")[0]
                loss = float(loss.split(".pth")[0])

                # Update the dictionary with the lowest loss for the current noise value
                if noise_value == desired_noise_level:
                    if (
                        noise_value not in lowest_losses
                        or loss < lowest_losses[noise_value][0]
                    ):
                        lowest_losses[noise_value] = (loss, file)

    # Check if the desired noise level was found and return the corresponding filename
    if desired_noise_level in lowest_losses:

        # Retrieve the lowest loss and associated filename for the desired noise level
        loss, file_name = lowest_losses[desired_noise_level]

        return file_name

    else:
        # Return None if no files match the desired noise level
        return None



@dataclass
class Viz:
    """
    A DataClass for handling various visualization settings and data.

    Attributes:
        dataset (Any): The dataset to visualize. Replace `Any` with the specific type.
        Printer (Optional[Type], optional): Printer for output. Defaults to None.
        verbose (bool, optional): Verbosity flag. Defaults to False.
        labelfigs_ (bool, optional): Flag to label figures. Defaults to True.
        SHO_ranges (Optional[Any], optional): Ranges for SHO data. Defaults to None.
        image_scalebar (Optional[Any], optional): Scalebar settings for images. Defaults to None.
        SHO_labels (List[Dict[str, str]], optional): Labels for SHO data. Defaults to predefined list.
        color_palette (Optional[Any], optional): Color palette settings. Defaults to None.

    """

    dataset: Any  # Specify the type based on what you expect
    # You can also define the type of Printer if you know it
    Printer: Optional[Type] = None
    verbose: bool = False
    labelfigs_: bool = True
    # Specify the type based on what you expect
    SHO_ranges: Optional[Any] = None
    # Specify the type based on what you expect
    image_scalebar: Optional[Any] = None

    SHO_labels: List[Dict[str, str]] = field(
        default_factory=lambda: [
            {"title": "Amplitude", "y_label": "Amplitude \n (Arb. U.)"},
            {"title": "Resonance Frequency", "y_label": "Resonance Frequency \n (Hz)"},
            {"title": "Dampening", "y_label": "Quality Factor \n (Arb. U.)"},
            {"title": "Phase", "y_label": "Phase \n (rad)"},
        ]
    )

    # Replace Any with the expected type if known
    color_palette: Optional[Any] = None

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
            current_state = args[0].dataset.get_state

            # Execute the decorated function and capture its output
            out = func(*args, **kwargs)

            # Restore the dataset's state to what it was before the function was called
            args[0].dataset.set_attributes(**current_state)

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
            current_dataset_state = self.dataset.get_state  # Assume this returns a dict of the dataset state

            # Debugging output to verify the preserved state
            print('current_SHO_ranges:', current_SHO_ranges)
            print('current_dataset_state:', current_dataset_state)

            # Call the original function with the given arguments
            out = func(self, SHO_data, *args, **kwargs)

            # Restore the preserved SHO_ranges
            self.SHO_ranges = current_SHO_ranges

            # Restore the preserved dataset state by setting the attributes back to their original values
            self.dataset.set_attributes(**current_dataset_state)

            return out

        return wrapper


    ##### GRAPHS #####

    @static_dataset_decorator
    def raw_data_comparison(
        self,
        true,
        predict=None,
        filename=None,
        pixel=None,
        voltage_step=None,
        legend=True,
        **kwargs,
    ):
        """
        Compare raw spectral data between true and predicted datasets.

        This function plots the real and imaginary components of the resampled data
        for a specified pixel and voltage step, allowing comparison between the true
        and predicted datasets. The function can save the plot if a filename is provided.

        Args:
            true (dict):
                Attributes of the true dataset to be set.
            predict (dict, optional):
                Attributes of the predicted dataset to be set. Defaults to None.
            filename (str, optional):
                Name of the file to save the figure. Defaults to None.
            pixel (int, optional):
                Pixel index to plot. If None, a random pixel is selected. Defaults to None.
            voltage_step (int, optional):
                Voltage step index to plot. If None, it is determined by the dataset. Defaults to None.
            legend (bool, optional):
                Whether to display a legend on the plot. Defaults to True.
            **kwargs (dict):
                Additional keyword arguments for the dataset's raw_spectra method.

        Returns:
            None
        """

        # Set the attributes for the true dataset
        self.set_attributes(**true)

        # Initialize figure and axes for plotting
        fig, axs = layout_fig(2, 2, figsize=(5, 1.25))

        # If a pixel is not provided, select a random pixel
        if pixel is None:
            pixel = np.random.randint(0, self.dataset.num_pix)

        # Get the voltage step, considering the current state
        voltage_step = self.get_voltage_step(voltage_step)

        # Set dataset state to grab the magnitude spectrum
        self.dataset.raw_format = "magnitude spectrum"

        # Get the raw spectral data for the selected pixel and voltage step
        data, x = self.dataset.raw_spectra(pixel, voltage_step, frequency=True)

        # Plot amplitude and phase for the true dataset
        axs[0].plot(x, data[0].flatten(), "b", label=self.dataset.label + " Amplitude")
        ax1 = axs[0].twinx()
        ax1.plot(x, data[1].flatten(), "r", label=self.dataset.label + " Phase")

        # If a predicted dataset is provided, plot its amplitude and phase
        if predict is not None:
            self.set_attributes(**predict)
            data, x = self.dataset.raw_spectra(
                pixel, voltage_step, frequency=True, **kwargs
            )
            axs[0].plot(
                x, data[0].flatten(), "bo", label=self.dataset.label + " Amplitude"
            )
            ax1.plot(x, data[1].flatten(), "ro", label=self.dataset.label + " Phase")
            self.set_attributes(**true)

        # Label the axes for the first subplot
        axs[0].set_xlabel("Frequency (Hz)")
        axs[0].set_ylabel("Amplitude (Arb. U.)")
        ax1.set_ylabel("Phase (rad)")

        # Reset dataset state to complex format
        self.dataset.raw_format = "complex"

        # Get the complex raw spectral data for the selected pixel and voltage step
        data, x = self.dataset.raw_spectra(pixel, voltage_step, frequency=True)

        # Plot real and imaginary components for the true dataset
        axs[1].plot(x, data[0].flatten(), "k", label=self.dataset.label + " Real")
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_ylabel("Real (Arb. U.)")
        ax2 = axs[1].twinx()
        ax2.set_ylabel("Imag (Arb. U.)")
        ax2.plot(x, data[1].flatten(), "g", label=self.dataset.label + " Imag")

        # If a predicted dataset is provided, plot its real and imaginary components
        if predict is not None:
            self.set_attributes(**predict)
            data, x = self.dataset.raw_spectra(
                pixel, voltage_step, frequency=True, **kwargs
            )
            axs[1].plot(x, data[0].flatten(), "ko", label=self.dataset.label + " Real")
            ax2.plot(x, data[1].flatten(), "gs", label=self.dataset.label + " Imag")
            self.set_attributes(**true)

        # Adjust the format of the tick labels and box aspect for all axes
        axes = [axs[0], axs[1], ax1, ax2]
        for ax in axes:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.set_box_aspect(1)

        # Optionally print the dataset states
        if self.verbose:
            print("True \n")
            self.set_attributes(**true)
            self.dataset.extraction_state
            if predict is not None:
                print("predicted \n")
                self.set_attributes(**predict)
                self.dataset.extraction_state

        # Display the legend if requested
        if legend:
            fig.legend(bbox_to_anchor=(1.0, 1), loc="upper right", borderaxespad=0.1)

        # Save the figure if a Printer object and filename are provided
        if self.Printer is not None and filename is not None:
            self.Printer.savefig(fig, filename, label_figs=[axs[0], axs[1]], style="b")

    @static_dataset_decorator
    def raw_be(
        self,
        dataset,
        x_start=0.8e6,
        x_end=1e6,
        figsize=(5 * (5 / 3), 1.3),
        inset_pos=[0.5, 0.65, 0.48, 0.33],
        filename="Figure_1_random_cantilever_resonance_results",
    ):
        """
        Plots the raw data and the Band Excitation (BE) waveform for a randomly selected 
        pixel and voltage step from the provided dataset. 

        This function performs the following steps:
        1. Selects a random pixel and voltage step from the dataset.
        2. Constructs and plots the BE waveform.
        3. Plots the resonance graph using the Fourier transform of the BE waveform.
        4. Plots the hysteresis waveform with a zoomed-in inset.
        5. Changes the dataset state to get the magnitude spectrum and plots it.
        6. Retrieves the raw spectra in both magnitude and complex format and plots the 
        real and imaginary components.
        7. Saves the figure if a printer object is available.

        Args:
            dataset (BE.dataset): BE dataset containing the data to be plotted.
            x_start (float, optional): Start of the x-axis range for the zoomed-in inset. Defaults to 0.8e6.
            x_end (float, optional): End of the x-axis range for the zoomed-in inset. Defaults to 1e6.
            figsize (tuple, optional): Size of the figure to be plotted. Defaults to (5 * (5 / 3), 1.3).
            inset_pos (list, optional): Position of the inset axes in the plot. Defaults to [0.5, 0.65, 0.48, 0.33].
            filename (str, optional): Name to save the file. Defaults to "Figure_1_random_cantilever_resonance_results".
        """
        
        # Select a random pixel and voltage step from the dataset to plot
        pixel = np.random.randint(0, dataset.num_pix)
        voltagestep = np.random.randint(0, dataset.voltage_steps)

        # Initialize the figure and axes for plotting
        fig, ax = layout_fig(5, 5, figsize=figsize)

        # Calculate the number of voltage steps in one BE waveform cycle
        be_voltagesteps = len(dataset.be_waveform) / dataset.be_repeats

        # Plot the BE waveform
        ax[0].plot(dataset.be_waveform[:int(be_voltagesteps)])
        ax[0].set(xlabel="Time (sec)", ylabel="Voltage (V)")

        # Perform Fourier Transform on the BE waveform to get the resonance graph
        resonance_graph = np.fft.fft(dataset.be_waveform[:int(be_voltagesteps)])
        fftfreq = fftpack.fftfreq(int(be_voltagesteps)) * dataset.sampling_rate

        # Plot the resonance graph
        ax[1].plot(
            fftfreq[:int(be_voltagesteps) // 2],
            np.abs(resonance_graph[:int(be_voltagesteps) // 2]),
        )
        ax[1].axvline(
            x=dataset.be_center_frequency,
            ymax=np.max(resonance_graph[:int(be_voltagesteps) // 2]),
            linestyle="--",
            color="r",
        )
        ax[1].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")

        # Set the x-axis limits based on the BE center frequency and bandwidth
        ax[1].set_xlim(
            dataset.be_center_frequency - dataset.be_bandwidth - dataset.be_bandwidth * 0.25,
            dataset.be_center_frequency + dataset.be_bandwidth + dataset.be_bandwidth * 0.25,
        )

        # Plot the hysteresis waveform and add a zoomed-in inset
        ax[2].plot(dataset.waveform_constructor())
        ax_new = ax[2].inset_axes(inset_pos)
        ax_new.plot(dataset.waveform_constructor())
        ax_new.set_xlim(x_start, x_end)
        ax_new.set_ylim(-2, 20)

        # Draw the inset connector lines
        inset_connector(
            fig,
            ax[2],
            ax_new,
            [(x_start, 0), (x_end, 0)],
            [(x_start, 0), (x_end, 0)],
            color="k",
            linestyle="--",
            linewidth=0.5,
        )

        # Add a box around the inset area on the main plot
        add_box(
            ax[2],
            (x_start, 0, x_end, 15),
            edgecolor="k",
            linestyle="--",
            facecolor="none",
            linewidth=0.5,
            zorder=10,
        )

        ax[2].set_xlabel("Voltage Steps")
        ax[2].set_ylabel("Voltage (V)")

        # Set the dataset state to retrieve the magnitude spectrum
        dataset.scaled = False
        dataset.raw_format = "magnitude spectrum"
        dataset.measurement_state = "all"
        dataset.resampled = False

        # Get the magnitude spectrum for the selected pixel and voltage step
        data_ = dataset.raw_spectra(pixel, voltagestep)

        # Plot the magnitude spectrum
        ax[3].plot(
            dataset.frequency_bin,
            data_[0].flatten(),
        )
        ax[3].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)", facecolor="none")

        # Plot the phase spectrum on the same plot with a secondary y-axis
        ax2 = ax[3].twinx()
        ax2.plot(
            dataset.frequency_bin,
            data_[1].flatten(),
            "r",
        )
        ax2.set(xlabel="Frequency (Hz)", ylabel="Phase (rad)")
        ax[3].set_zorder(ax2.get_zorder() + 1)

        # Switch the dataset back to complex format
        dataset.raw_format = "complex"
        data_ = dataset.raw_spectra(pixel, voltagestep)

        # Plot the real and imaginary components of the spectra
        ax[4].plot(dataset.frequency_bin, data_[0].flatten(), label="Real")
        ax[4].set(xlabel="Frequency (Hz)", ylabel="Real (Arb. U.)")
        ax3 = ax[4].twinx()
        ax3.plot(dataset.frequency_bin, data_[1].flatten(), "r", label="Imaginary")
        ax3.set(xlabel="Frequency (Hz)", ylabel="Imag (Arb. U.)", facecolor="none")

        # Save the figure if a Printer object is available
        if self.Printer is not None:
            self.Printer.savefig(fig, filename, label_figs=ax, style="b")
            
    @static_scale_decorator
    def SHO_hist(self, SHO_data, filename=None, scaled=False):
        """Plots the SHO hysteresis parameters

        Args:
            SHO_data (numpy): SHO fit results
            filename (str, optional): filename where to save the results. Defaults to "".
        """

        # if the scale is False will not use the scale in the viz
        if self.dataset.scaled or scaled:
            print('dataset is scaled')
            self.SHO_ranges = None

        # if the SHO data is not a list it will make it a list
        if type(SHO_data) is not list:
            SHO_data = [SHO_data]

        # check distributions of each parameter before and after scaling
        fig, axs = layout_fig(
            4 * len(SHO_data), 4, figsize=(5.25, 1.25 * len(SHO_data))
        )

        for k, SHO_data_ in enumerate(SHO_data):
            axs_ = axs[k * 4: (k + 1) * 4]

            SHO_data_ = SHO_data_.reshape(-1, 4)

            for i, (ax, label) in enumerate(zip(axs_.flat, self.SHO_labels)):
                ax.hist(
                    SHO_data_[:, i].flatten(),
                    100,
                    range=self.SHO_ranges[i] if self.SHO_ranges else None,
                )

                if i == 0:
                    ax.set(ylabel="counts")
                ax.set(xlabel=label["y_label"])
                ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

                ax.xaxis.labelpad = 10

                ax.set_box_aspect(1)

            if self.verbose:
                self.dataset.extraction_state

        # prints the figure
        if self.Printer is not None and filename is not None:
            self.Printer.savefig(fig, filename, label_figs=axs, style="b")

    def SHO_loops(self, data=None, filename="Figure_2_random_SHO_fit_results"):
        """
        Plots the SHO loop fit results for a randomly selected pixel or provided data.

        Args:
            data (np.array, optional): The dataset to use for plotting the SHO loop fits. 
                                    If not provided, data from a randomly selected pixel is used. Defaults to None.
            filename (str, optional): The filename for saving the plotted figure. 
                                    Defaults to "Figure_2_random_SHO_fit_results".

        This function selects a pixel either randomly or based on the provided data and 
        plots the SHO (Simple Harmonic Oscillator) loop fit results across various 
        parameters (defined in self.SHO_labels). The resulting plot is saved using 
        the specified filename if a Printer object is available.
        """

        if data is None:
            # If no data is provided, select a random pixel from the dataset
            pixel = np.random.randint(0, self.dataset.num_pix)
            data = self.dataset.SHO_fit_results()[[pixel], :, :]

        # Initialize the figure and axes with a 4x4 grid layout
        fig, axs = layout_fig(4, 4, figsize=(5.5, 1.1))

        # Loop over each axis and corresponding SHO label to plot the fit results
        for i, (ax, label) in enumerate(zip(axs, self.SHO_labels)):
            ax.plot(self.dataset.dc_voltage, data[0, :, i])
            ax.set_ylabel(label["y_label"])

        # If verbose mode is enabled, log the current extraction state (for debugging or tracking)
        if self.verbose:
            self.dataset.extraction_state

        # If a Printer object is defined, save the figure with the specified filename and style
        if self.Printer is not None:
            self.Printer.savefig(fig, filename, label_figs=axs, style="b")
            
    @static_dataset_decorator
    def fit_tester(self, true, predict, pixel=None, voltage_step=None, **kwargs):
        """
        Tests the fit of a model by comparing predicted data against true data for a specific pixel and voltage step.

        If a pixel is not provided, a random pixel will be selected. The method will also determine the appropriate 
        voltage step if one is not provided. The comparison is visualized using a raw data comparison plot.

        Args:
            true (dict): A dictionary containing the true data values to compare against.
            predict (dict): A dictionary containing the predicted data values.
            pixel (int, optional): The pixel index to use for the comparison. If not provided, a random pixel will be selected.
            voltage_step (int, optional): The voltage step to use for the comparison. If not provided, it will be calculated based on the current state.
            **kwargs: Additional keyword arguments passed to the raw_data_comparison method.

        Returns:
            None
        """

        # If a pixel is not provided, select a random pixel from the dataset
        if pixel is None:
            pixel = np.random.randint(0, self.dataset.num_pix)

        # Get the appropriate voltage step, considering the current state
        voltage_step = self.get_voltage_step(voltage_step)

        # Set object attributes based on the predict dictionary
        self.set_attributes(**predict)

        # Compute the fit parameters for the selected pixel and voltage step
        params = self.dataset.SHO_LSQF(pixel=pixel, voltage_step=voltage_step)

        # Print the true data for inspection or debugging
        print(true)

        # Perform and visualize the raw data comparison between true and predicted data
        self.raw_data_comparison(
            true,
            predict,
            pixel=pixel,
            voltage_step=voltage_step,
            fit_results=params,
            **kwargs,
        )

    @static_dataset_decorator
    def nn_checker(
        self, state, filename=None, pixel=None, voltage_step=None, legend=True, **kwargs
    ):
        # if a pixel is not provided it will select a random pixel
        if pixel is None:
            # Select a random point and time step to plot
            pixel = np.random.randint(0, self.dataset.num_pix)

        # gets the voltagestep with consideration of the current state
        voltage_step = self.get_voltage_step(voltage_step)

        self.set_attributes(**state)

        data = self.dataset.raw_spectra(pixel=pixel, voltage_step=voltage_step)

        # plot real and imaginary components of resampled data
        fig = plt.figure(figsize=(3, 1.25), layout="compressed")
        axs = plt.subplot(111)

        self.dataset.raw_format = "complex"

        data, x = self.dataset.raw_spectra(
            pixel, voltage_step, frequency=True, **kwargs
        )

        axs.plot(x, data[0].flatten(), "k", label=self.dataset.label + " Real")
        axs.set_xlabel("Frequency (Hz)")
        axs.set_ylabel("Real (Arb. U.)")
        ax2 = axs.twinx()
        ax2.set_ylabel("Imag (Arb. U.)")
        ax2.plot(x, data[1].flatten(), "g", label=self.dataset.label + " Imag")

        axes = [axs, ax2]

        for ax in axes:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.set_box_aspect(1)

        if self.verbose:
            self.dataset.extraction_state

        if legend:
            fig.legend(bbox_to_anchor=(1.0, 1),
                       loc="upper right", borderaxespad=0.1)

        # prints the figure
        if self.Printer is not None and filename is not None:
            self.Printer.savefig(fig, filename, style="b")
    
    ##### Analytics #####
    
    @static_dataset_decorator
    def bmw_nn(
        self,
        true_state,
        prediction=None,
        model=None,
        out_state=None,
        n=1,
        gaps=(0.8, 0.33),
        size=(1.25, 1.25),
        filename=None,
        compare_state=None,
        fit_type='SHO',
        **kwargs,
    ):
        
        #TODO: I had to remove this for fitting --
        # true_state = torch.atleast_3d(torch.tensor(true_state.reshape(-1,96)))
        
        d1, d2, x1, x2, label, index1, mse1 = None, None, None, None, None, None, None

        if fit_type == "SHO":
            d1, d2, x1, x2, label, index1, mse1 = self.get_best_median_worst(
                true_state,
                prediction=prediction,
                model=model,
                out_state=out_state,
                n=n,
                compare_state=compare_state,
                **kwargs,
            )

            fig, ax = subfigures(1, 3, gaps=gaps, size=size)

            for i, (true, prediction, error) in enumerate(zip(d1, d2, mse1)):
                ax_ = ax[i]
                ax_.plot(
                    x2,
                    prediction[0].flatten(),
                    color_palette["NN_A"],
                    label=f"NN {label[0]}",
                )
                ax1 = ax_.twinx()
                ax1.plot(
                    x2,
                    prediction[1].flatten(),
                    color_palette["NN_P"],
                    label=f"NN {label[1]}]",
                )

                ax_.plot(
                    x1,
                    true[0].flatten(),
                    "o",
                    color=color_palette["NN_A"],
                    label=f"Raw {label[0]}",
                )
                ax1.plot(
                    x1,
                    true[1].flatten(),
                    "o",
                    color=color_palette["NN_P"],
                    label=f"Raw {label[1]}",
                )

                ax_.set_xlabel("Frequency (Hz)")

                # Position text at (1 inch, 2 inches) from the bottom left corner of the figure
                text_position_in_inches = (
                    -1 * (gaps[0] + size[0]) * ((2 - i) % 3) + size[0] / 2,
                    (gaps[1] + size[1]) * (1.25 - i // 3 - 1.25) - gaps[1],
                )
                text = f"MSE: {error:0.4f}"
                add_text_to_figure(
                    fig, text, text_position_in_inches, fontsize=6, ha="center"
                )

                if out_state is not None:
                    if "measurement state" in out_state.keys():
                        if out_state["raw_format"] == "magnitude spectrum":
                            ax_.set_ylabel("Amplitude (Arb. U.)")
                            ax1.set_ylabel("Phase (rad)")
                    else:
                        ax_.set_ylabel("Real (Arb. U.)")
                        ax1.set_ylabel("Imag (Arb. U.)")

            # add a legend just for the last one
            lines, labels = ax_.get_legend_handles_labels()
            lines2, labels2 = ax1.get_legend_handles_labels()
            ax_.legend(lines + lines2, labels + labels2, loc="upper right")

        elif fit_type == "hysteresis":
            d1, d2, x1, x2, index1, mse1, _ = self.get_best_median_worst_hysteresis(
                true_state,
                prediction=prediction,
                n=n,
                **kwargs,
            )

            fig, ax = subfigures(1, 3, gaps=gaps, size=size)

            for i, (true, prediction, error) in enumerate(zip(d1, d2, mse1)):
                ax_ = ax[i]

                ax_.plot(
                    x2,
                    prediction.flatten(),
                    color=color_palette["NN_A"],
                    # label=f"NN {label[0]}",
                )

                ax_.plot(
                    x1,
                    true.flatten(),
                    "o",
                    color=color_palette["NN_A"],
                    # label=f"Raw {label[0]}",
                )

                ax_.set_xlabel("Voltage (V)")

                # Position text at (1 inch, 2 inches) from the bottom left corner of the figure
                text_position_in_inches = (
                    -1 * (gaps[0] + size[0]) * ((2 - i) % 3) + size[0] / 2,
                    (gaps[1] + size[1]) * (1.25 - i // 3 - 1.25) - gaps[1],
                )

                text = f"MSE: {error:0.4f}"
                add_text_to_figure(
                    fig, text, text_position_in_inches, fontsize=6, ha="center"
                )

                ax_.set_ylabel("(Arb. U.)")

                # add a legend just for the last one
                lines, labels = ax_.get_legend_handles_labels()
                ax_.legend(lines, labels, loc="upper right")

        else:
            raise ValueError("fit_type must be SHO or hysteresis")

        # prints the figure
        if self.Printer is not None and filename is not None:
            self.Printer.savefig(fig, filename, label_figs=ax, style="b")

        if "returns" in kwargs.keys():
            if kwargs["returns"] == True:
                return d1, d2, index1, mse1
    
    @static_dataset_decorator
    def SHO_switching_maps(
        self,
        SHO_,
        colorbars=True,
        clims=[
            (0, 1.4e-4),  # amplitude
            (1.31e6, 1.33e6),  # resonance frequency
            (-230, -160),  # quality factor
            (-np.pi, np.pi),  # phase
        ],  # phase limits
        measurement_state="off",  # sets the measurement state to extract the data
        cycle=2,  # cycle number to extract
        cols=3,  # number of columns in the plot grid
        fig_width=6.5,  # width of the figure in inches
        number_of_steps=9,  # number of voltage steps to display
        voltage_plot_height=1.25,  # height of the voltage plot in inches
        intra_gap=0.02,  # gap between individual plots in inches
        inter_gap=0.05,  # gap between plot rows in inches
        cbar_gap=0.5,  # gap between colorbars in inches
        cbar_space=1.3,  # space reserved for colorbars on the right
        filename=None,  # optional filename to save the figure
    ):
        """
        Generates a plot of switching maps for SHO data (Amplitude, Resonance Frequency, Quality Factor, Phase)
        across multiple voltage steps.

        Args:
            SHO_ (torch.Tensor or np.ndarray): SHO data containing amplitude, resonance frequency, quality factor, and phase.
            colorbars (bool): If True, adds colorbars to the plots. Defaults to True.
            clims (list): List of tuples representing color limits for each type of data (Amplitude, Resonance Frequency, 
                        Quality Factor, Phase). Defaults are provided.
            measurement_state (str): State of the measurement to get the data ('on' or 'off'). Defaults to "off".
            cycle (int): The measurement cycle number to extract the data from. Defaults to 2.
            cols (int): Number of columns in the plot grid. Defaults to 3.
            fig_width (float): Width of the figure in inches. Defaults to 6.5.
            number_of_steps (int): Number of voltage steps to display. Defaults to 9.
            voltage_plot_height (float): Height of the voltage plot in inches. Defaults to 1.25.
            intra_gap (float): Gap between individual plots in inches. Defaults to 0.02.
            inter_gap (float): Gap between plot rows in inches. Defaults to 0.05.
            cbar_gap (float): Gap between colorbars in inches. Defaults to 0.5.
            cbar_space (float): Space reserved on the right for colorbars in inches. Defaults to 1.3.
            filename (str, optional): If provided, saves the figure to the specified filename. Defaults to None.

        Returns:
            fig (matplotlib.figure.Figure): The generated figure containing the switching maps.
        """
        
        # Set the measurement state and cycle in the dataset
        self.dataset.measurement_state = measurement_state
        self.dataset.cycle = cycle

        # Initialize the list for storing the axes
        ax = []

        # Calculate the number of rows for the plot grid
        rows = np.ceil(number_of_steps / 3)

        # Calculate the size of the individual image embeddings in the figure
        embedding_image_size = (
            fig_width - (inter_gap * (cols - 1)) - intra_gap * 3 * cols - cbar_space * colorbars
        ) / (cols * 4)

        # Calculate the total height of the figure
        fig_height = (
            rows * (embedding_image_size + inter_gap) + voltage_plot_height + 0.33
        )

        # Convert figure dimensions to relative coordinates for axes positioning
        fig_scalar = FigDimConverter((fig_width, fig_height))

        # Create the figure with the specified dimensions
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Define the position and size of the voltage plot
        pos_inch = [
            0.33,
            fig_height - voltage_plot_height,
            fig_width - 0.33,
            voltage_plot_height,
        ]

        # Add the voltage plot to the figure
        ax.append(fig.add_axes(fig_scalar.to_relative(pos_inch)))

        # Reset the position for embedding plots
        pos_inch[0] = 0
        pos_inch[1] -= embedding_image_size + 0.33

        # Set the size for each embedding plot
        pos_inch[2] = embedding_image_size
        pos_inch[3] = embedding_image_size

        # Add embedding plots to the figure for each voltage step
        for i in range(number_of_steps):
            for j in range(4):  # Amplitude, Resonant Frequency, Quality Factor, Phase
                ax.append(fig.add_axes(fig_scalar.to_relative(pos_inch)))
                pos_inch[0] += embedding_image_size + intra_gap

            # Move to the next row if necessary
            if (i + 1) % cols == 0 and i != 0:
                pos_inch[0] = 0
                pos_inch[1] -= embedding_image_size + inter_gap
            else:
                pos_inch[0] += inter_gap

        # Retrieve the DC voltage data from the dataset
        voltage = self.dataset.dc_voltage

        # Select a specific cycle from the dataset, if applicable
        if hasattr(self.dataset, "cycle") and self.dataset.cycle is not None:
            voltage = self.dataset.get_cycle(voltage)

        # Get indices of the voltage steps to plot
        inds = np.linspace(0, len(voltage) - 1, number_of_steps, dtype=int)

        # Convert SHO_ data to numpy if it's a PyTorch tensor
        if isinstance(SHO_, torch.Tensor):
            SHO_ = SHO_.detach().numpy()

        # Reshape SHO_ data to match the required format
        SHO_ = SHO_.reshape(self.dataset.num_pix, self.dataset.voltage_steps, 4)

        # Get the specific measurement cycle from the dataset
        SHO_ = self.dataset.get_measurement_cycle(SHO_, axis=1)

        # Plot the voltage data
        ax[0].plot(voltage, "k")
        ax[0].set_ylabel("Voltage (V)")
        ax[0].set_xlabel("Step")

        # Add markers and labels for each voltage step
        for i, ind in enumerate(inds):
            ax[0].plot(ind, voltage[ind], "o", color="k", markersize=10)
            vshift = (ax[0].get_ylim()[1] - ax[0].get_ylim()[0]) * 0.25

            # Adjust label position if necessary
            if voltage[ind] - vshift - 0.15 < ax[0].get_ylim()[0]:
                vshift = -vshift / 2

            # Add step number labels to the voltage plot
            ax[0].text(ind, voltage[ind] - vshift, str(i + 1), color="k", fontsize=12)

        # Data names for each of the four properties
        names = ["A", "\u03C9", "Q", "\u03C6"]

        # Plot amplitude, resonant frequency, quality factor, and phase data
        for i, ind in enumerate(inds):
            for j in range(4):
                imagemap(
                    ax[i * 4 + j + 1],
                    SHO_[:, ind, j],
                    colorbars=False,
                    cmap="viridis",
                )

                # Label figures if in the first row
                if i // rows == 0:
                    labelfigs(
                        ax[i * 4 + j + 1],
                        string_add=names[j],
                        loc="cb",
                        size=5,
                        inset_fraction=(0.2, 0.2),
                    )

                # Set color limits for the plot
                ax[i * 4 + j + 1].images[0].set_clim(clims[j])

            # Add step number labels to the plots
            labelfigs(
                ax[1::4][i],
                string_add=str(i + 1),
                size=5,
                loc="bl",
                inset_fraction=(0.2, 0.2),
            )

        # Add colorbars to the plots if enabled
        if colorbars:
            bar_ax = []
            voltage_ax_pos = fig_scalar.to_inches(np.array(ax[0].get_position()).flatten())

            for i in range(4):
                # Calculate position and size of colorbars
                cbar_h = (voltage_ax_pos[1] - inter_gap - 2 * intra_gap - 0.33) / 2
                cbar_w = (cbar_space - inter_gap - 2 * cbar_gap) / 2
                pos_inch = [
                    voltage_ax_pos[2] - (2 - i % 2) * (cbar_gap + cbar_w) + inter_gap,
                    voltage_ax_pos[1] - (i // 2) * (inter_gap + cbar_h) - 0.33 - cbar_h,
                    cbar_w,
                    cbar_h,
                ]

                # Add colorbar to the figure
                bar_ax.append(fig.add_axes(fig_scalar.to_relative(pos_inch)))
                cbar = plt.colorbar(ax[i + 1].images[0], cax=bar_ax[i], format="%.1e")
                cbar.set_label(names[i])  # Add label to the colorbar

        # Save the figure if a filename is provided
        if self.Printer is not None and filename is not None:
            self.Printer.savefig(
                fig, filename, size=6, loc="tl", inset_fraction=(0.2, 0.2)
            )

        # Show the figure
        fig.show()
        
    @static_dataset_decorator
    def SHO_Fit_comparison(
        self,
        data,
        names,
        gaps=(0.8, 0.9),
        size=(1.25, 1.25),
        model_comparison=None,
        out_state=None,
        filename=None,
        display_results="all",
        **kwargs,
    ):
        """
        Generates a comparison plot of SHO (Simple Harmonic Oscillator) fit results.

        This function creates subplots comparing multiple fits (e.g., LSQF, NN) for amplitude and phase of 
        cantilever responses. It supports comparing multiple fit models, visualizing the predicted and true 
        responses, and optionally displaying error metrics like Mean Squared Error (MSE) for each fit.

        Args:
            data (list): List of tuples, where each tuple contains data for comparison, including:
                        - d1: true amplitude
                        - d2: predicted amplitude
                        - x1: true frequency points
                        - x2: predicted frequency points
                        - label: labels for amplitude and phase
                        - index1: index of the dataset
                        - mse1: Mean Squared Error values
                        - params: fit parameters (SHO)
            names (list): List of strings representing the names of the fits (e.g., "LSQF", "NN").
            gaps (tuple, optional): Tuple defining gaps between subplots. Defaults to (0.8, 0.9).
            size (tuple, optional): Tuple defining the size of each subplot. Defaults to (1.25, 1.25).
            model_comparison (list, optional): List of additional models (e.g., neural networks or LSQF fits) to compare.
                                            Defaults to None.
            out_state (dict, optional): Dictionary defining the output format and other parameters. Defaults to None.
            filename (str, optional): If provided, saves the figure to this filename. Defaults to None.
            display_results (str, optional): Controls the type of results displayed (e.g., MSE, all). Defaults to "all".
            **kwargs: Additional keyword arguments.

        Returns:
            matplotlib.figure.Figure: The generated figure containing the SHO fit comparison plots.

        Notes:
            - This function plots the raw and predicted amplitude and phase data for each model.
            - It supports displaying detailed error metrics for amplitude, phase, frequency, and quality factor.
            - The function supports saving the generated figure to a file using the `Printer` object.
        """

        # Get the number of fits from the length of the data list
        num_fits = len(data)

        # Adjust gaps based on the type of results to display (e.g., only MSE)
        if display_results == "MSE":
            gaps = (0.8, 0.45)
        elif display_results is None:
            gaps = (0.8, 0.33)

        # Create subplots for the comparison
        fig, ax = subfigures(3, num_fits, gaps=gaps, size=size)

        # Loop through each fit and the associated data
        for step, (data, name) in enumerate(zip(data, names)):
            # Unpack the data (true, predicted values, indices, etc.)
            d1, d2, x1, x2, label, index1, mse1, params = data

            # Loop through datasets for comparison (true vs. predicted data)
            for bmw, (true, prediction, error, SHO, index1) in enumerate(zip(d1, d2, mse1, params, index1)):
                # Initialize dictionaries for errors and SHO parameters
                errors = {}
                SHOs = {}

                # Determine the subplot index
                i = bmw * num_fits + step
                ax_ = ax[i]

                # Plot predicted amplitude and phase
                ax_.plot(
                    x2,
                    prediction[0].flatten(),
                    color=color_palette[f"{name}_A"],
                    label=f"{name} {label[0]}",
                )
                ax1 = ax_.twinx()
                ax1.plot(
                    x2,
                    prediction[1].flatten(),
                    color=color_palette[f"{name}_P"],
                    label=f"{name} {label[1]}",
                )

                # Plot true amplitude and phase
                ax_.plot(
                    x1,
                    true[0].flatten(),
                    "o",
                    color=color_palette["LSQF_A"],
                    label=f"Raw {label[0]}",
                )
                ax1.plot(
                    x1,
                    true[1].flatten(),
                    "o",
                    color=color_palette["LSQF_P"],
                    label=f"Raw {label[1]}",
                )

                # Store errors and SHO parameters for the current model
                errors[name] = error
                SHOs[name] = SHO

                # If a model comparison is provided, plot the comparison results
                if model_comparison is not None:
                    if model_comparison[step] is not None:
                        # Get SHO parameters from the comparison model
                        pred_data, params, labels = self.get_SHO_params(
                            index1, model=model_comparison[step], out_state=out_state
                        )

                        # Determine the color prefix based on model type (NN or LSQF)
                        if isinstance(model_comparison[step], nn.Module):
                            color = "NN"
                        elif isinstance(model_comparison[step], dict):
                            color = "LSQF"

                        # Store errors and SHO parameters for the comparison model
                        errors[color] = self.get_mse_index(index1, model_comparison[step])
                        SHOs[color] = np.array(params).squeeze()

                        # Plot the comparison data
                        ax_.plot(
                            x2,
                            pred_data.squeeze()[0].flatten(),
                            color=color_palette[f"{color}_A"],
                            label=f"{color} {labels[0]}",
                        )
                        ax1.plot(
                            x2,
                            pred_data.squeeze()[1].flatten(),
                            color=color_palette[f"{color}_P"],
                            label=f"{color} {labels[1]}",
                        )

                        # Display detailed results if requested
                        if display_results == "all":
                            error_string = f"MSE - LSQF: {errors['LSQF']:0.4f} NN: {errors['NN']:0.4f}\n AMP - LSQF: {SHOs['LSQF'][0]:0.2e} NN: {SHOs['NN'][0]:0.2e}\n\u03C9 - LSQF: {SHOs['LSQF'][1]/1000:0.1f} NN: {SHOs['NN'][1]/1000:0.1f} Hz\nQ - LSQF: {SHOs['LSQF'][2]:0.1f} NN: {SHOs['NN'][2]:0.1f}\n\u03C6 - LSQF: {SHOs['LSQF'][3]:0.2f} NN: {SHOs['NN'][3]:0.1f} rad"
                        elif display_results == "MSE":
                            error_string = f"MSE - LSQF: {errors['LSQF']:0.4f} NN: {errors['NN']:0.4f}"

                # Set the x-axis label (Frequency in Hz)
                ax_.set_xlabel("Frequency (Hz)")

                # Display the results (e.g., MSE) below the plots
                if display_results is not None:
                    center = get_axis_pos_inches(fig, ax[i])
                    text_position_in_inches = (center[0], center[1] - 0.33)

                    if "error_string" not in locals():
                        error_string = f"MSE: {error:0.4f}"

                    add_text_to_figure(
                        fig,
                        error_string,
                        text_position_in_inches,
                        fontsize=6,
                        ha="center",
                        va="top",
                    )

                # Set y-axis labels based on output state
                if out_state is not None:
                    if "raw_format" in out_state.keys() and out_state["raw_format"] == "magnitude spectrum":
                        ax_.set_ylabel("Amplitude (Arb. U.)")
                        ax1.set_ylabel("Phase (rad)")
                    else:
                        ax_.set_ylabel("Real (Arb. U.)")
                        ax1.set_ylabel("Imag (Arb. U.)")

                # Add legend for the last fit
                if i < num_fits:
                    lines, labels = ax_.get_legend_handles_labels()
                    lines2, labels2 = ax1.get_legend_handles_labels()
                    ax_.legend(lines + lines2, labels + labels2, loc="upper right")

        # Save the figure if filename is provided
        if self.Printer is not None and filename is not None:
            self.Printer.savefig(fig, filename, label_figs=ax, style="b")


    
    ###### MOVIES #####

    @static_dataset_decorator
    def SHO_fit_movie_images(
        self,
        noise=0,
        model_path=None,
        models=[None],
        fig_width=6.5,
        voltage_plot_height=1.25,  # height of the voltage plot
        intra_gap=0.02,  # gap between the graphs
        inter_gap=0.2,  # gap between the graphs
        cbar_gap=0.6,  # gap between the graphs of colorbars
        cbar_space=1.3,  # space on the right where the colorbar is not
        colorbars=True,
        scalebar_=True,
        filename=None,
        basepath=None,
        labels=None,
        phase_shift=None,
    ):
        """
        Generates a sequence of images depicting SHO (Simple Harmonic Oscillator) fit results
        for various voltage steps, and optionally compiles them into a movie.

        This function creates images showing the fit results of the SHO model for both the 
        "on" and "off" states at different voltage steps. The images can include multiple 
        models for comparison, and optional features like colorbars, scalebars, and labels. 
        The images are saved to the specified directory, and a movie can be created from them.

        Args:
            noise (int, optional): The noise level used for generating the SHO fits. Defaults to 0.
            model_path (str, optional): Path to the directory containing the model checkpoints. Defaults to None.
            models (list, optional): List of models to compare. Defaults to [None].
            fig_width (float, optional): Width of the figure. Defaults to 6.5.
            voltage_plot_height (float, optional): Height of the voltage plot. Defaults to 1.25.
            intra_gap (float, optional): Gap between the graphs of the same dataset. Defaults to 0.02.
            inter_gap (float, optional): Gap between the graphs of different datasets. Defaults to 0.2.
            cbar_gap (float, optional): Gap between the graphs and colorbars. Defaults to 0.6.
            cbar_space (float, optional): Space reserved for the colorbars on the right. Defaults to 1.3.
            colorbars (bool, optional): Whether to include colorbars in the images. Defaults to True.
            scalebar_ (bool, optional): Whether to include a scalebar in the images. Defaults to True.
            filename (str, optional): Base filename for saving images. Defaults to None.
            basepath (str, optional): Base path for saving images. Defaults to None.
            labels (list, optional): Labels for the different models in the comparison. Defaults to None.
            phase_shift (list, optional): Phase shifts to apply to the models. Defaults to None.

        Returns:
            None: The function saves the generated images and optionally creates a movie from them.
        """

        # Sets the output state to ensure the dataset outputs pixel data
        output_state = {"output_shape": "pixels", "scaled": False}
        self.dataset.set_attributes(**output_state)

        # Constructs the basepath for saving images if provided
        if basepath is not None:
            # If a model path is provided, name the directory based on the model with the lowest loss
            if model_path is not None:
                model_filename = (
                    model_path
                    + "/"
                    + get_lowest_loss_for_noise_level(model_path, noise)
                )
                basepath += f"/{model_filename.split('/')[-1].split('.')[0]}"
            else:
                # If no model is provided, name the directory based on the noise level
                basepath += f"Noise_{noise}"

            # Creates the directory for saving images
            basepath = make_folder(basepath)

        # If models are provided for comparison
        if models is not None:
            on_data = []
            off_data = []
            noise_labels = []

            # Loop through the models and get the SHO data for each
            for model_, phase_shift_ in zip(models, phase_shift):
                on_models, off_models = self.get_SHO_data(
                    noise, model_, phase_shift=phase_shift_
                )
                on_data.append(on_models)
                off_data.append(off_models)
                noise_labels.append(noise)
        else:
            # If no models are provided, get the default model and its SHO data
            model = self.get_model(model_path, noise)
            on_data, off_data = self.get_SHO_data(noise, model)

        # Labels for the different SHO parameters (e.g., Amplitude, Frequency, Quality Factor, Phase)
        names = ["A", "\u03C9", "Q", "\u03C6"]

        # Retrieves the DC voltage data (only for the "on" state)
        voltage = self.dataset.dc_voltage

        # Loop through each voltage step to generate images
        for z, voltage in enumerate(voltage):
            # Build the figure and axes layout for the movie images
            fig, ax, fig_scalar = self.build_figure_for_movie(
                models,  # dataset to compare to
                fig_width,  # width of the figure
                inter_gap,  # gap between the graphs of different datasets
                intra_gap,  # gap between the graphs of the same datasets
                cbar_space,  # gap between the graphs and the colorbar
                colorbars,  # include colorbars or not
                voltage_plot_height,  # height of the voltage plot
                labels,  # labels for the models
            )

            # Plot the DC voltage trace for the current step
            ax[0].plot(self.dataset.dc_voltage, "k")
            ax[0].plot(z, voltage, "o", color="k", markersize=10)
            ax[0].set_ylabel("Voltage (V)")
            ax[0].set_xlabel("Step")

            # Loop over the models and SHO parameters to plot the images
            for compare_num in range(len(models)):
                for j in range(4):
                    # Plot each SHO parameter for the "on" state
                    imagemap(
                        ax[j + 1 + compare_num * 8],
                        on_data[compare_num][:, z, j],
                        colorbars=False,
                        clim=self.SHO_ranges[j],
                    )
                    # Plot each SHO parameter for the "off" state
                    imagemap(
                        ax[j + 5 + compare_num * 8],
                        off_data[compare_num][:, z, j],
                        colorbars=False,
                        clim=self.SHO_ranges[j],
                    )
                    labelfigs(ax[j + 1], string_add=f"On {names[j]}", loc="ct")
                    labelfigs(ax[j + 5], string_add=f"Off {names[j]}", loc="ct")

                # Add labels to the figures if provided
                if labels is not None:
                    # Get the position of the axis
                    bbox = ax[5 + compare_num * 8].get_position()

                    # Calculate the position for the label text
                    top_in_norm_units = bbox.bounds[1] + bbox.bounds[3]
                    right_in_norm_units = bbox.bounds[0] + bbox.bounds[2]

                    # Convert to inches
                    fig_size_inches = fig.get_size_inches()
                    fig_height_inches = fig_size_inches[1]
                    fig_width_inches = fig_size_inches[0]

                    top_in_inches = top_in_norm_units * fig_height_inches
                    right_in_inches = right_in_norm_units * fig_width_inches + inter_gap

                    # Add the label text to the figure
                    add_text_to_figure(
                        fig,
                        f"{labels[compare_num]} Noise {noise_labels[compare_num]}",
                        [right_in_inches / 2, top_in_inches + 0.33 / 2],
                    )

                # Add colorbars if specified
                if colorbars:
                    bar_ax = []

                    # Get the voltage axis position in inches
                    voltage_ax_pos = fig_scalar.to_inches(
                        np.array(ax[0].get_position()).flatten()
                    )

                    # Loop through the 4 SHO parameters to add colorbars
                    for i in range(4):
                        # Calculate the position and size of the colorbars
                        cbar_h = (voltage_ax_pos[1] - inter_gap * 2 - 0.33) / 2
                        cbar_w = (cbar_space - inter_gap - cbar_gap) / 2

                        pos_inch = [
                            voltage_ax_pos[2]
                            - (2 - i % 2) * (cbar_gap + cbar_w)
                            + inter_gap
                            + cbar_w,
                            voltage_ax_pos[1]
                            - (i // 2) * (inter_gap + cbar_h)
                            - 0.33
                            - cbar_h,
                            cbar_w,
                            cbar_h,
                        ]

                        # Add the colorbar axis to the figure
                        bar_ax.append(fig.add_axes(
                            fig_scalar.to_relative(pos_inch)))

                        # Add the colorbar to the axis
                        cbar = plt.colorbar(
                            ax[i + 1].images[0],
                            cax=bar_ax[i],
                            format="%.1e",
                            ticks=np.linspace(
                                self.SHO_ranges[i][0], self.SHO_ranges[i][1], 5
                            ),
                        )

                        cbar.set_label(names[i])  # Label the colorbar

            # Add a scalebar to the last axis if specified
            if self.image_scalebar is not None:
                scalebar(ax[-1], *self.image_scalebar)

            # Save the figure if a Printer object and filename are provided
            if self.Printer is not None and filename is not None:
                self.Printer.savefig(
                    fig,
                    f"{filename}_noise_{noise}_{z:04d}",
                    basepath=basepath + "/",
                    fileformats=["png"],
                )

            plt.close(fig)  # Close the figure to free memory

        # Create a movie from the saved images
        make_movie(
            f"{filename}_noise_{noise}", basepath, basepath, file_format="png", fps=5
        )

    def build_figure_for_movie(
        self,
        comparison,
        fig_width,
        inter_gap,
        intra_gap,
        cbar_space,
        colorbars,
        voltage_plot_height,
        labels=None,
    ):
        """
        Builds a figure layout for generating movie frames with multiple comparison plots.

        This function creates a figure layout that includes multiple rows and columns of 
        subplots, which are used to display comparison data (e.g., SHO fit results) alongside 
        a voltage plot. It is designed to accommodate various configurations, including optional 
        colorbars, labels, and gaps between plots.

        Args:
            comparison (any): Dataset(s) to compare, which determine the number of rows in the figure.
            fig_width (float): Width of the figure in inches.
            inter_gap (float): Gap between different datasets in inches.
            intra_gap (float): Gap between similar datasets in inches.
            cbar_space (float): Space allocated for the colorbar in inches.
            colorbars (bool): Whether to include colorbars in the figure.
            voltage_plot_height (float): Height of the voltage plot in inches.
            labels (list, optional): List of labels for the plots. Defaults to None.

        Returns:
            matplotlib.figure.Figure: The figure object containing the plots.
            list: A list of matplotlib.axes.Axes objects for each subplot.
            FigDimConverter: An object used to convert figure dimensions from inches to relative coordinates.
        """

        # Initialize the list of axes for the figure
        ax = []

        # Calculate the number of rows needed, based on the comparison datasets
        rows = len(comparison) * 2

        # Determine the number of inter-gaps based on whether labels are provided
        if labels is not None:
            inter_gap_count = len(comparison) + 1
        else:
            inter_gap_count = 1

        # Calculate the size of each embedding image in the figure
        embedding_image_size = (
            fig_width
            - inter_gap * inter_gap_count
            - intra_gap * 2
            - cbar_space * colorbars
        ) / 4  # Divide by 4 because there are 4 plots per row

        # Calculate the total figure height based on the image sizes and gaps
        fig_height = (
            rows * (embedding_image_size + inter_gap / 2 + intra_gap / 2)
            + voltage_plot_height
            + 0.33 * inter_gap_count
        )

        # Create a scalar to convert inches to relative coordinates for positioning
        fig_scalar = FigDimConverter((fig_width, fig_height))

        # Create the figure with the calculated width and height
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Define the position for the voltage plot (left, bottom, width, height in inches)
        pos_inch = [
            0.33,  # Left position
            fig_height - voltage_plot_height,  # Bottom position (top-aligned)
            6.5 - 0.33,  # Width of the voltage plot
            voltage_plot_height,  # Height of the voltage plot
        ]

        # Add the voltage plot to the figure
        ax.append(fig.add_axes(fig_scalar.to_relative(pos_inch)))

        # Reset the x position for embedding plots and adjust the y position
        pos_inch[0] = 0  # Reset left position
        pos_inch[1] -= embedding_image_size + 0.33 * inter_gap_count  # Adjust bottom position

        # Set the size for embedding images
        pos_inch[2] = embedding_image_size  # Width of embedding image
        pos_inch[3] = embedding_image_size  # Height of embedding image

        # Loop through the rows to add the subplots for the embedding images
        for j in range(rows):
            # Add 4 graphs per row
            for i in range(4):
                ax.append(fig.add_axes(fig_scalar.to_relative(pos_inch)))  # Add subplot to figure

                # Adjust the gap between plots within the same row
                if i == 1:
                    gap = inter_gap
                else:
                    gap = intra_gap

                # Move the position to the right for the next subplot
                pos_inch[0] += embedding_image_size + gap

            # Reset the x position to the start of the next row
            pos_inch[0] = 0

            # Adjust the y position for the next row based on the row index
            if (j + 1) % 2 == 0:
                pos_inch[1] -= embedding_image_size + inter_gap * inter_gap_count
            else:
                pos_inch[1] -= embedding_image_size + intra_gap

        # Create a reordered list of axes for easier access
        ax_ = [ax[0]]  # Start with the voltage plot

        z = len(comparison) - 1

        # Reorder the axes to make them easier to work with, going left to right, top to bottom
        for j in range(1 + z):
            for i in range(2):
                ax_.extend(ax[1 + 2 * i + 8 * j: 3 + 2 * i + 8 * j])
                ax_.extend(ax[5 + 2 * i + 8 * j: 7 + 2 * i + 8 * j])

        return fig, ax_, fig_scalar


#     def get_model(self, model_path, noise):
#         # if a model path is not provided then data is from LSQF
#         if model_path is not None:
#             # finds the model with the lowest loss for the given noise level
#             model_filename = (
#                 model_path + "/" +
#                 get_lowest_loss_for_noise_level(model_path, noise)
#             )

#             # instantiate the model
#             model = SHO_Model(
#                 self.dataset, training=False, model_basename="SHO_Fitter_original_data"
#             )

#             # loads the weights
#             model.load(model_filename)

#             if self.verbose:
#                 # prints which model is being used
#                 print("Using model: ", model_filename)

#         elif model is not None:
#             pass

#         else:
#             # sets the model equal to None if no model is provided
#             model = None

#         return model


    ##### GETTERS #####

    def get_voltage_step(self, voltage_step):
        """
        Determine and return a valid voltage step index.

        This method checks if a voltage step index is provided. If not, it randomly
        selects a valid voltage step index based on the current measurement state of
        the dataset.

        Args:
            voltage_step (int, optional):
                The voltage step index to use. If None, a random index is selected based
                on the dataset's measurement state.

        Returns:
            int:
                The selected or provided voltage step index.
        """

        # If voltage_step is not provided, determine a random step
        if voltage_step is None:
            # If the measurement state is "on" or "off", select from the first half of the steps
            if (
                self.dataset.measurement_state == "on"
                or self.dataset.measurement_state == "off"
            ):
                voltage_step = np.random.randint(0, self.dataset.voltage_steps // 2)
            else:
                # Otherwise, select from the full range of voltage steps
                voltage_step = np.random.randint(0, self.dataset.voltage_steps)

        # Return the determined or provided voltage step index
        return voltage_step
    
    def get_SHO_data(self, noise, model, phase_shift=None):
        """
        Retrieves Simple Harmonic Oscillator (SHO) fit results for both "on" and "off" states.

        This function sets the noise level and measurement state of the dataset, then extracts 
        the SHO fit results for both the "on" and "off" states using the specified model and 
        phase shift.

        Args:
            noise (int): The noise level to apply to the dataset.
            model (object): The model used for generating the SHO fit results.
            phase_shift (float, optional): An optional phase shift to apply during the fit. Defaults to None.

        Returns:
            tuple: A tuple containing two numpy arrays:
                - `on_data`: SHO fit results for the "on" state.
                - `off_data`: SHO fit results for the "off" state.
        """

        # Set the noise level for the dataset
        self.dataset.noise = noise

        # Set the measurement state to "on" to get the data for the "on" state
        self.dataset.measurement_state = "on"

        # Retrieve the SHO fit results for the "on" state
        on_data = self.dataset.SHO_fit_results(
            model=model, phase_shift=phase_shift
        )

        # Set the measurement state to "off" to get the data for the "off" state
        self.dataset.measurement_state = "off"

        # Retrieve the SHO fit results for the "off" state
        off_data = self.dataset.SHO_fit_results(
            model=model, phase_shift=phase_shift
        )

        # Return the fit results for both states as a tuple
        return on_data, off_data
    
    @static_dataset_decorator
    def get_SHO_params(self, index, model, out_state):
        """
        Retrieves Simple Harmonic Oscillator (SHO) parameters for a given index based on the specified model.

        This function computes or retrieves the SHO parameters (such as amplitude, phase, resonance frequency, and quality factor)
        for the provided indices using either a neural network model or an LSQF model, depending on the type of `model` provided.
        It also processes the data based on the output state specified in `out_state`.

        Args:
            index (list): List of indices for which to retrieve the SHO parameters.
            model (any): The model used to compute the SHO results. Can be a neural network (`nn.Module`) or a dictionary representing
                        an LSQF model with specific parameters.
            out_state (dict): Dictionary specifying the output state of the data, such as how the output should be formatted.

        Returns:
            np.array, np.array, list: 
                - `pred_data`: The predicted SHO data (processed real/imaginary or amplitude/phase data).
                - `params`: The corresponding SHO parameters (e.g., amplitude, phase, resonance frequency, quality factor).
                - `labels`: A list of labels describing the parameters for the returned data.
        """

        # Get pixel and voltage coordinates from the provided indices
        pixel, voltage = np.unravel_index(
            index, (self.dataset.num_pix, self.dataset.voltage_steps)
        )

        # Case 1: The model is a neural network (nn.Module)
        if isinstance(model, nn.Module):
            # Retrieve the input data for the neural network
            X_data, Y_data = self.dataset.NN_data()

            # Select the data based on the provided indices
            X_data = X_data[[index]]

            # Use the model to predict the data and SHO parameters
            pred_data, scaled_param, params = model.predict(X_data)

            # Convert the predicted data to a NumPy array
            pred_data = np.array(pred_data)

        # Case 2: The model is a dictionary (assumed to be an LSQF model)
        if isinstance(model, dict):
            # Ensure that the dataset is not scaled when retrieving raw parameters
            self.dataset.scaled = False

            # Retrieve the SHO fit results without any phase shift
            params_shifted = self.dataset.SHO_fit_results()

            # Ensure the phase shift for the current fitter is set to zero
            exec(f"self.dataset.{model['fitter']}_phase_shift = 0")

            # Retrieve the SHO fit parameters
            params = self.dataset.SHO_fit_results()

            # Switch back to scaled parameters for further processing
            self.dataset.scaled = True

            # Generate raw spectra from the fit results
            pred_data = self.dataset.raw_spectra(fit_results=params)

            # Reshape the predicted data for correct dimensionality (samples, channels, voltage steps)
            pred_data = np.array([pred_data[0], pred_data[1]])  # (channels, samples, voltage steps)
            pred_data = np.swapaxes(pred_data, 0, 1)  # (samples, channels, voltage steps)
            pred_data = np.swapaxes(pred_data, 1, 2)  # (samples, voltage steps, channels)

            # Reshape the shifted parameters for consistent handling
            params_shifted = params_shifted.reshape(-1, 4)

            # Select the data and parameters based on the provided indices
            pred_data = pred_data[[index]]
            params = params_shifted[[index]]

        # Swap axes of the predicted data to match expected output format
        pred_data = np.swapaxes(pred_data, 1, 2)

        # Apply output state processing to the predicted data (real/imaginary or amplitude/phase)
        pred_data, labels = self.out_state(pred_data, out_state)

        # Return the predicted data, SHO parameters, and their corresponding labels
        return pred_data, params, labels

    ###### SETTERS ######

    def set_attributes(self, **kwargs):
        """
        Sets the attributes of the dataset using key-value pairs from a dictionary.

        This utility function iterates over the provided keyword arguments and sets
        the corresponding attributes of the dataset object. It also ensures that any
        necessary setters are triggered, such as for the 'noise' attribute.

        Args:
            **kwargs:
                Arbitrary keyword arguments representing the attributes to set on the dataset.
                The keys represent attribute names, and the values represent the values to be set.

        Returns:
            None
        """

        # Iterate over the key-value pairs in kwargs and set the corresponding attributes on the dataset
        for key, value in kwargs.items():
            setattr(self.dataset, key, value)

        # Ensure that the setter for 'noise' is called if the 'noise' attribute is provided in kwargs
        if kwargs.get("noise"):
            self.noise = kwargs.get("noise")








#     def get_freq_values(self, data):
#         data = data.flatten()
#         if len(data) == self.dataset.resampled_bins:
#             x = resample(self.dataset.frequency_bin,
#                          self.dataset.resampled_bins)
#         elif len(data) == len(self.dataset.frequency_bin):
#             x = self.dataset.frequency_bin
#         else:
#             raise ValueError(
#                 "original data must be the same length as the frequency bins or the resampled frequency bins"
#             )
#         return x


#     @static_dataset_decorator
#     def nn_validation(
#         self,
#         model,
#         data=None,
#         unscaled=True,
#         pixel=None,
#         voltage_step=None,
#         index=None,
#         legend=True,
#         filename=None,
#         **kwargs,
#     ):
#         # Makes the figure
#         fig, axs = layout_fig(2, 2, figsize=(5, 1.25))

#         # sets the dataset state to grab the magnitude spectrum
#         state = {"raw_format": "magnitude spectrum", "resampled": True}
#         self.set_attributes(**state)

#         # if set to scaled it will change the label
#         if unscaled:
#             label = ""
#         else:
#             label = "scaled"

#         # if an index is not provided it will select a random index
#         # it is also possible to use a voltage step
#         if index is None:
#             # if a voltage step is provided it will use the voltage step to grab a specific index
#             if voltage_step is not None:
#                 # if a pixel is not provided it will select a random pixel
#                 if pixel is None:
#                     # Select a random point and time step to plot
#                     pixel = np.random.randint(0, self.dataset.num_pix)

#                 # gets the voltagestep with consideration of the current state
#                 voltage_step = self.get_voltage_step(voltage_step)

#                 # gets the data based on a specific pixel and voltagestep
#                 data, x = self.dataset.raw_spectra(
#                     pixel, voltage_step, frequency=True, **kwargs
#                 )

#                 SHO_results = self.dataset.SHO_LSQF(pixel, voltage_step)

#         # if a smaller manual dataset is provided it will use that
#         if data is not None:
#             # if an index is not provided it will select a random index
#             if index is None:
#                 index = np.random.randint(0, data.shape[0])

#             # grabs the data based on the index
#             data = data[[index]]

#             # gets the frequency values from the dataset
#             x = self.get_freq_values(data[:, :, 0])

#         # computes the prediction from the nn model
#         pred_data, scaled_param, parm = model.predict(data)

#         # unscales the data
#         if unscaled:
#             data_complex = self.dataset.raw_data_scaler.inverse_transform(data)
#             pred_data_complex = self.dataset.raw_data_scaler.inverse_transform(
#                 pred_data.numpy()
#             )
#         else:
#             data_complex = data
#             pred_data_complex = self.dataset.to_complex(pred_data.numpy())

#         # computes the magnitude spectrum from the data
#         data_magnitude = self.dataset.to_magnitude(data_complex)
#         pred_data_magnitude = self.dataset.to_magnitude(pred_data_complex)

#         # plots the data
#         axs[0].plot(
#             x,
#             pred_data_magnitude[0].flatten(),
#             "b",
#             label=label + " Amplitude \n NN Prediction",
#         )
#         ax1 = axs[0].twinx()
#         ax1.plot(
#             x,
#             pred_data_magnitude[1].flatten(),
#             "r",
#             label=label + " Phase \n NN Prediction",
#         )

#         axs[0].plot(x, data_magnitude[0].flatten(),
#                     "bo", label=label + " Amplitude")
#         ax1.plot(x, data_magnitude[1].flatten(), "ro", label=label + " Phase")

#         axs[0].set_xlabel("Frequency (Hz)")
#         axs[0].set_ylabel("Amplitude (Arb. U.)")
#         ax1.set_ylabel("Phase (rad)")

#         pred_data_complex = self.dataset.to_real_imag(pred_data_complex)
#         data_complex = self.dataset.to_real_imag(data_complex)

#         axs[1].plot(
#             x,
#             pred_data_complex[0].flatten(),
#             "k",
#             label=label + " Real \n NN Prediction",
#         )
#         axs[1].set_xlabel("Frequency (Hz)")
#         axs[1].set_ylabel("Real (Arb. U.)")
#         ax2 = axs[1].twinx()
#         ax2.set_ylabel("Imag (Arb. U.)")
#         ax2.plot(
#             x,
#             pred_data_complex[1].flatten(),
#             "g",
#             label=label + " Imag \n NN Prediction",
#         )

#         axs[1].plot(x, data_complex[0].flatten(), "ko", label=label + " Real")
#         ax2.plot(x, data_complex[1].flatten(), "gs", label=label + " Imag")

#         axes = [axs[0], axs[1], ax1, ax2]

#         for ax in axes:
#             ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
#             ax.set_box_aspect(1)

#         if legend:
#             fig.legend(bbox_to_anchor=(1.0, 1),
#                        loc="upper right", borderaxespad=0.1)

#         if "SHO_results" in kwargs:
#             if voltage_step is None:
#                 SHO_results = kwargs["SHO_results"][[index]]

#                 if unscaled:
#                     SHO_results = self.dataset.SHO_scaler.inverse_transform(
#                         SHO_results)

#             SHO_results = SHO_results.squeeze()

#             fig.text(
#                 0.5,
#                 -0.05,
#                 f"LSQF: A = {SHO_results[0]:0.2e} ,  \u03C9 = {SHO_results[1]/1000:0.1f} Hz, Q = {SHO_results[2]:0.1f}, \u03C6 = {SHO_results[3]:0.2f} rad",
#                 ha="center",
#                 fontsize=6,
#             )

#             parm = parm.detach().numpy().squeeze()

#             fig.text(
#                 0.5,
#                 -0.15,
#                 f"NN: A = {parm[0]:0.2e} ,  \u03C9 = {parm[1]/1000:0.1f} Hz, Q = {parm[2]:0.1f}, \u03C6 = {parm[3]:0.2f} rad",
#                 ha="center",
#                 fontsize=6,
#             )

#         # prints the figure
#         if self.Printer is not None and filename is not None:
#             self.Printer.savefig(fig, filename, label_figs=[
#                                  axs[0], axs[1]], style="b")

#     @static_dataset_decorator
#     def best_median_worst_reconstructions(
#         self,
#         model,
#         true,
#         SHO_values=None,
#         labels=["NN", "LSQF"],
#         unscaled=True,
#         filename=None,
#         **kwargs,
#     ):
#         gaps = (0.8, 0.9)
#         size = (1.25, 1.25)

#         fig, ax = subfigures(3, 3, gaps=gaps, size=size)

#         dpi = fig.get_dpi()

#         prediction, scaled_param, parm = model.predict(true)

#         index, mse = model.mse_rankings(true, prediction)

#         ind = np.hstack(
#             (index[0:3], index[len(index) // 2 -
#              1: len(index) // 2 + 2], index[-3:])
#         )
#         mse = np.hstack(
#             (mse[0:3], mse[len(index) // 2 - 1: len(index) // 2 + 2], mse[-3:])
#         )

#         x = self.get_freq_values(true[0, :, 0])

#         # unscales the data
#         if unscaled:
#             true = self.dataset.raw_data_scaler.inverse_transform(true)
#             prediction = self.dataset.raw_data_scaler.inverse_transform(
#                 prediction.numpy()
#             )

#             if self.dataset.raw_format == "magnitude spectrum":
#                 # computes the magnitude spectrum from the data
#                 true = self.dataset.to_magnitude(true)
#                 prediction = self.dataset.to_magnitude(prediction)
#             else:
#                 # computes the magnitude spectrum from the data
#                 true = self.dataset.to_real_imag(true)
#                 prediction = self.dataset.to_real_imag(prediction)

#             SHO_values = self.dataset.SHO_scaler.inverse_transform(SHO_values)

#         else:
#             true = true
#             prediction = self.dataset.to_complex(prediction.numpy())

#         ax.reverse()

#         for i, (ind_, ax_) in enumerate(zip(ind, ax)):
#             ax_.plot(
#                 x, prediction[0][ind_].flatten(), "b", label=labels[0] + " Amplitude"
#             )
#             ax1 = ax_.twinx()
#             ax1.plot(x, prediction[1][ind_].flatten(),
#                      "r", label=labels[0] + " Phase")

#             ax_.plot(x, true[0][ind_].flatten(), "bo", label="Raw Amplitude")
#             ax1.plot(x, true[1][ind_].flatten(), "ro", label="Raw Phase")

#             ax_.set_xlabel("Frequency (Hz)")
#             ax_.set_ylabel("Amplitude (Arb. U.)")
#             ax1.set_ylabel("Phase (rad)")

#             # Position text at (1 inch, 2 inches) from the bottom left corner of the figure
#             text_position_in_inches = (
#                 (gaps[0] + size[0]) * (i % 3),
#                 (gaps[1] + size[1]) * (3 - i // 3 - 1) - 0.8,
#             )
#             text = f"MSE: {mse[i]:0.4f}\nA LSQF:{SHO_values[ind_, 0]:0.2e} NN:{parm[ind_, 0]:0.2e}\n\u03C9: LSQF: {SHO_values[ind_, 1]/1000:0.1f} NN: {parm[ind_, 1]/1000:0.1f} Hz\nQ: LSQF: {SHO_values[ind_, 2]:0.1f} NN: {parm[ind_, 2]:0.1f}\n\u03C6: LSQF: {SHO_values[ind_, 3]:0.2f} NN: {parm[ind_, 3]:0.1f} rad"
#             add_text_to_figure(fig, text, text_position_in_inches, fontsize=6)

#             if i == 2:
#                 lines, labels = ax_.get_legend_handles_labels()
#                 lines2, labels2 = ax1.get_legend_handles_labels()
#                 ax_.legend(lines + lines2, labels + labels2, loc="upper right")

#         # prints the figure
#         if self.Printer is not None and filename is not None:
#             self.Printer.savefig(fig, filename, style="b")
    
    #TODO add comments and docstring
    def get_best_median_worst(
        self,
        true_state,
        prediction=None,
        out_state=None,
        n=1,
        SHO_results=False,
        index=None,
        compare_state=None,
        **kwargs,
    ):
        def data_converter(data):
            
            # converts to a standard form which is a list
            data = self.dataset.to_real_imag(data)

            try:
                # converts to numpy from tensor
                data = [data.numpy() for data in data]
            except:
                pass

            return data

        if type(true_state) is dict:
            self.set_attributes(**true_state)

            # the data must be scaled to rank the results
            self.dataset.scaled = True

            true, x1 = self.dataset.raw_spectra(frequency=True)

        # condition if x_data is passed
        else:

            true = data_converter(true_state)
        
            # gets the frequency values
            if true[0].ndim == 2:
                x1 = self.dataset.get_freq_values(true[0].shape[1])

        # holds the raw state
        current_state = self.dataset.get_state

        if isinstance(prediction, nn.Module):
            fitter = "NN"

            # sets the phase shift to zero for parameters
            # This is important if doing the fits because the fits will be wrong if the phase is shifted.
            self.dataset.NN_phase_shift = 0

            data = self.dataset.to_nn(true)

            pred_data, scaled_params, params = prediction.predict(data)

            self.dataset.scaled = True

            prediction, x2 = self.dataset.raw_spectra(
                fit_results=params, frequency=True
            )

        elif isinstance(prediction, dict):
            fitter = prediction["fitter"]

            exec(f"self.dataset.{prediction['fitter']}_phase_shift =0")

            self.dataset.scaled = False

            params = self.dataset.SHO_fit_results()

            params = params.reshape(-1, 4)

            self.dataset.scaled = True

            prediction, x2 = self.dataset.raw_spectra(
                fit_results=params, frequency=True
            )

        if "x2" not in locals():
            # if you do not use the model will run the
            x2 = self.dataset.get_freq_values(prediction[0].shape[1])

        # index the data if provided
        if index is not None:
            true = [true[0][index], true[1][index]]
            prediction = [prediction[0][index], prediction[1][index]]
            # params = params[index]

        if compare_state is not None:
            compare_state = data_converter(compare_state)

            # this must take the scaled data
            index1, mse1, d1, d2 = get_rankings(
                compare_state, prediction, n=n
            )
        else:
            # this must take the scaled data
            index1, mse1, d1, d2 = get_rankings(
                true, prediction, n=n)

        d1, labels = self.out_state(d1, out_state)
        d2, labels = self.out_state(d2, out_state)

        # saves just the parameters that are needed
        params = params[index1]

        # resets the current state to apply the phase shifts
        self.set_attributes(**current_state)

        # gets the original index values
        if index is not None:
            index1 = index[index1]

        # if statement that will return the values for the SHO Results
        if SHO_results:
            if eval(f"self.dataset.{fitter}_phase_shift") is not None:
                params[:, 3] = eval(
                    f"self.dataset.shift_phase(params[:, 3], self.dataset.{fitter}_phase_shift)"
                )
            return (d1, d2, x1, x2, labels, index1, mse1, params)
        else:
            return (d1, d2, x1, x2, labels, index1, mse1)

    #TODO: add comments and docstring
    def out_state(self, data, out_state):
        # holds the raw state
        current_state = self.dataset.get_state

        def convert_to_mag(data):
            data = self.dataset.to_complex(data, axis=1)
            data = self.dataset.raw_data_scaler.inverse_transform(data)
            data = self.dataset.to_magnitude(data)
            data = np.array(data)
            data = np.rollaxis(data, 0, data.ndim - 1)
            return data

        labels = ["real", "imaginary"]

        if out_state is not None:
            if "raw_format" in out_state.keys():
                if out_state["raw_format"] == "magnitude spectrum":
                    data = convert_to_mag(data)
                    labels = ["Amplitude", "Phase"]

            elif "scaled" in out_state.keys():
                if out_state["scaled"] == False:
                    data = self.dataset.raw_data_scaler.inverse_transform(data)
                    labels = ["Scaled " + s for s in labels]

        self.set_attributes(**current_state)

        return data, labels

#     @static_dataset_decorator
#     def get_best_median_worst_hysteresis(self,
#                                          true_state,
#                                          prediction=None,
#                                         #  out_state=None,
#                                          n=1,
#                                         index=None,
#                                          **kwargs):

#         true = true_state
#         x1 = self.dataset.get_voltage

#         data = torch.tensor(true).float()

#         if isinstance(prediction, Fitter1D.Model):
#             pred_data, scaled_params, params = prediction.predict(
#                 data, translate_params=False, is_SHO=False)
#         elif isinstance(prediction, np.ndarray):
#             pred_data = prediction
#         else:
#             raise ValueError("prediction must be a Model or a numpy array")

#         prediction = pred_data

#         x2 = self.dataset.get_voltage

#         # # index the data if provided
#         # if index is not None:
#         #     true = [true[0][index], true[1][index]]
#         #     prediction = [prediction[0][index], prediction[1][index]]

#         # converts to numpy from tensor if needed
#         try:
#             prediction = prediction.detach().numpy()
#             true = true.detach().numpy()
#         except:
#             pass

#         prediction = np.rollaxis(prediction, 0, prediction.ndim - 1)
#         true = np.rollaxis(true, 0, true.ndim - 1)

#         # this must take the scaled data
#         index1, mse1, d1, d2 = Model.get_rankings(
#             true, prediction, n=n)

#         # saves the parameters if the model is provided
#         try:
#             # saves just the parameters that are needed
#             params = params[index1]
#         except:
#             params = None

#         # gets the original index values
#         if index is not None:
#             index1 = index[index1]

#         return (d1, d2, x1, x2, index1, mse1, params)

#     @static_dataset_decorator
#     def get_mse_index(self, index, model):
#         # gets the raw data
#         # returns the raw spectra in (samples, voltage steps, real/imaginary)
#         data, _ = self.dataset.NN_data()

#         # gets the index of the data selected
#         # (samples, voltage steps, real/imaginary)
#         data = data[[index]]

#         if isinstance(model, nn.Module):
#             # gets the predictions from the neural network
#             predictions, params_scaled, params = model.predict(data)

#             # detaches the tensor and converts to numpy
#             predictions = predictions.detach().numpy()

#         if isinstance(model, dict):
#             # # holds the raw state
#             # current_state = self.dataset.get_state

#             # sets the phase shift to zero for the specific fitter - this is a requirement for using the fitting function
#             exec(f"self.dataset.{model['fitter']}_phase_shift =0")

#             # Ensures that we get the unscaled parameters
#             # Only the unscaled parameters can be used to calculate the raw data
#             self.dataset.scaled = False

#             # Gets the parameters
#             params = self.dataset.SHO_fit_results()

#             # sets the dataset to scaled
#             # we compare the MSE using the scaled parameters
#             self.dataset.scaled = True

#             # Ensures that the measurement state is complex
#             self.dataset.raw_format = "complex"

#             # This returns the raw data based on the parameters
#             # this returns a list of the real and imaginary data
#             pred_data = self.dataset.raw_spectra(fit_results=params)

#             # makes the data an array
#             # (real/imaginary, samples, voltage steps)
#             pred_data = np.array(pred_data)

#             # rolls the axis to (samples, voltage steps, real/imaginary)
#             pred_data = np.rollaxis(pred_data, 0, pred_data.ndim)

#             # gets the index of the data selected
#             # (samples, voltage steps, real/imaginary)
#             predictions = pred_data[[index]]

#             # # restores the state to the original state
#             # self.set_attributes(**current_state)

#         return SHO_Model.MSE(data.detach().numpy(), predictions)




#     @static_dataset_decorator
#     def SHO_switching_maps_test(
#         self,
#         SHO_,
#         colorbars=True,
#         clims=[
#             (0, 1.4e-4),  # amplitude
#             (1.31e6, 1.33e6),  # resonance frequency
#             (-230, -160),  # quality factor
#             (-np.pi, np.pi),
#         ],  # phase
#         measurement_state="off",  # sets the measurement state to get the data
#         cycle=2,  # sets the cycle to get the data
#         cols=3,
#         fig_width=6.5,  # figure width in inches
#         number_of_steps=9,  # number of steps on the graph
#         voltage_plot_height=1.25,  # height of the voltage plot
#         intra_gap=0.02,  # gap between the graphs,
#         inter_gap=0.05,  # gap between the graphs,
#         cbar_gap=0.5,  # gap between the graphs of colorbars
#         cbar_space=1.3,  # space on the right where the cbar is not
#         filename=None,
#         labels=None,
#     ):
#         if type(SHO_) is not list:
#             SHO_ = [SHO_]

#         comp_number = len(SHO_)

#         # sets the voltage state to off, and the cycle to get
#         self.dataset.measurement_state = measurement_state
#         self.dataset.cycle = cycle

#         # instantiates the list of axes
#         ax = []

#         # number of rows
#         rows = np.ceil(number_of_steps * comp_number / 3)

#         # calculates the size of the embedding image
#         embedding_image_size = (
#             fig_width
#             - (inter_gap * (cols - 1))
#             - intra_gap * 3 * cols
#             - cbar_space * colorbars
#         ) / (cols * 4)

#         # calculates the figure height based on the image details
#         fig_height = (
#             rows * (embedding_image_size + inter_gap)
#             + voltage_plot_height
#             + 0.33
#             + inter_gap * (comp_number - 1)
#         )

#         # defines a scalar to convert inches to relative coordinates
#         fig_scalar = FigDimConverter((fig_width, fig_height))

#         # creates the figure
#         fig = plt.figure(figsize=(fig_width, fig_height))

#         # left bottom width height
#         pos_inch = [
#             0.33,
#             fig_height - voltage_plot_height,
#             6.5 - 0.33,
#             voltage_plot_height,
#         ]

#         # adds the plot for the voltage
#         ax.append(fig.add_axes(fig_scalar.to_relative(pos_inch)))

#         # resets the x0 position for the embedding plots
#         pos_inch[0] = 0
#         pos_inch[1] -= embedding_image_size + 0.33

#         # sets the embedding size of the image
#         pos_inch[2] = embedding_image_size
#         pos_inch[3] = embedding_image_size

#         # This makes the figures
#         for k, _SHO in enumerate(SHO_):
#             # adds the embedding plots
#             for i in range(number_of_steps):
#                 # loops around the amp, phase, and freq
#                 for j in range(4):
#                     # adds the plot to the figure
#                     ax.append(fig.add_axes(fig_scalar.to_relative(pos_inch)))

#                     # adds the inter plot gap
#                     pos_inch[0] += embedding_image_size + intra_gap

#                 # if the last column in row, moves the position to the next row
#                 if (i + 1) % cols == 0 and i != 0:
#                     # resets the x0 position for the embedding plots
#                     pos_inch[0] = 0

#                     # moves the y0 position to the next row
#                     pos_inch[1] -= embedding_image_size + inter_gap

#                     if (i + 1) % (cols * comp_number) == 0 and comp_number > 1:
#                         pos_inch[1] -= inter_gap

#                 else:
#                     # adds the small gap between the plots
#                     pos_inch[0] += inter_gap

#         # gets the DC voltage data - this is for only the on state or else it would all be 0
#         voltage = self.dataset.dc_voltage

#         # gets just part of the loop
#         if hasattr(self.dataset, "cycle") and self.dataset.cycle is not None:
#             # gets the cycle of interest
#             voltage = self.dataset.get_cycle(voltage)

#         # gets the index of the voltage steps to plot
#         inds = np.linspace(0, len(voltage) - 1, number_of_steps, dtype=int)

#         # plots the voltage
#         ax[0].plot(voltage, "k")
#         ax[0].set_ylabel("Voltage (V)")
#         ax[0].set_xlabel("Step")

#         # Plot the data with different markers
#         for i, ind in enumerate(inds):
#             # this adds the labels to the graphs
#             ax[0].plot(ind, voltage[ind], "o", color="k", markersize=10)
#             vshift = (ax[0].get_ylim()[1] - ax[0].get_ylim()[0]) * 0.25

#             # positions the location of the labels
#             if voltage[ind] - vshift - 0.15 < ax[0].get_ylim()[0]:
#                 vshift = -vshift / 2

#             # adds the text to the graphs
#             ax[0].text(ind, voltage[ind] - vshift,
#                        str(i + 1), color="k", fontsize=12)

#         for k, _SHO in enumerate(SHO_):
#             # converts the data to a numpy array
#             if isinstance(_SHO, torch.Tensor):
#                 _SHO = _SHO.detach().numpy()

#             print(_SHO.shape)
#             _SHO = _SHO.reshape(self.dataset.num_pix,
#                                 self.dataset.voltage_steps, 4)

#             # get the selected measurement cycle
#             _SHO = self.dataset.get_measurement_cycle(_SHO, axis=1)

#             names = ["A", "\u03C9", "Q", "\u03C6"]

#             for i, ind in enumerate(inds):
#                 axis_start = int(
#                     (i % cols) * 4
#                     + ((i) // cols) * (comp_number * cols * 4)
#                     + k * (cols * 4)
#                     + 1
#                 )

#                 # loops around the amp, resonant frequency, and Q, Phase
#                 for j in range(4):
#                     imagemap(
#                         ax[axis_start + j],
#                         _SHO[:, ind, j],
#                         colorbars=False,
#                         cmap="viridis",
#                     )

#                     if i // rows == 0 and k == 0:
#                         labelfigs(
#                             ax[axis_start + j],
#                             string_add=names[j],
#                             loc="cb",
#                             size=5,
#                             inset_fraction=(0.2, 0.2),
#                         )

#                     ax[axis_start + j].images[0].set_clim(clims[j])

#                     if k == 0:
#                         labelfigs(
#                             ax[axis_start + j],
#                             string_add=str(i + 1),
#                             size=5,
#                             loc="bl",
#                             inset_fraction=(0.2, 0.2),
#                         )

#                     if (axis_start + j) % (4 * cols) == 1:
#                         ax[axis_start + j].set_ylabel(labels[k])

#         # if add colorbars
#         if colorbars:
#             # builds a list to store the colorbar axis objects
#             bar_ax = []

#             # gets the voltage axis position in ([xmin, ymin, xmax, ymax]])
#             voltage_ax_pos = fig_scalar.to_inches(
#                 np.array(ax[0].get_position()).flatten()
#             )

#             # loops around the 4 axis
#             for i in range(4):
#                 # calculates the height and width of the colorbars
#                 cbar_h = (voltage_ax_pos[1] -
#                           inter_gap - 2 * intra_gap - 0.33) / 2
#                 cbar_w = (cbar_space - inter_gap - 2 * cbar_gap) / 2

#                 # sets the position of the axis in inches
#                 pos_inch = [
#                     voltage_ax_pos[2] - (2 - i % 2) *
#                     (cbar_gap + cbar_w) + inter_gap,
#                     voltage_ax_pos[1] - (i // 2) *
#                     (inter_gap + cbar_h) - 0.33 - cbar_h,
#                     cbar_w - 0.02,
#                     cbar_h - 0.1,
#                 ]

#                 # adds the plot to the figure
#                 bar_ax.append(fig.add_axes(fig_scalar.to_relative(pos_inch)))

#                 # adds the colorbars to the plots
#                 cbar = plt.colorbar(ax[i + 1].images[0],
#                                     cax=bar_ax[i], format="%.1e")
#                 cbar.set_label(names[i])  # Add a label to the colorbar

#         # prints the figure
#         if self.Printer is not None and filename is not None:
#             self.Printer.savefig(
#                 fig, filename, size=6, loc="tl", inset_fraction=(0.2, 0.2)
#             )

#         fig.show()

#     @static_dataset_decorator
#     def noisy_datasets(
#         self, state, noise_level=None, pixel=None, voltage_step=None, filename=None
#     ):
#         if pixel is None:
#             # Select a random point and time step to plot
#             pixel = np.random.randint(0, self.dataset.num_pix)

#         if voltage_step is None:
#             voltage_step = np.random.randint(0, self.dataset.voltage_steps)

#         self.set_attributes(**state)

#         if noise_level is None:
#             datasets = np.arange(0, len(self.dataset.raw_datasets))
#         else:
#             datasets = noise_level

#         fig, ax_ = layout_fig(
#             len(datasets),
#             4,
#             figsize=(4 * (1.25 + 0.33), ((1 + len(datasets)) // 4) * 1.25),
#         )

#         for i, (ax, noise) in enumerate(zip(ax_, datasets)):
#             self.dataset.noise = noise

#             data, x = self.dataset.raw_spectra(
#                 pixel, voltage_step, frequency=True)

#             ax.plot(x, data[0].flatten(), color="k")
#             ax1 = ax.twinx()
#             ax1.plot(x, data[1].flatten(), color="b")

#             ax.set_xlabel("Frequency (Hz)")

#             if self.dataset.raw_format == "magnitude spectrum":
#                 ax.set_ylabel("Amplitude (Arb. U.)")
#                 ax1.set_ylabel("Phase (rad)")
#             elif self.dataset.raw_format == "complex":
#                 ax.set_ylabel("Real (Arb. U.)")
#                 ax1.set_ylabel("Imag (Arb. U.)")

#             # makes the box square
#             ax.set_box_aspect(1)

#             labelfigs(
#                 ax1,
#                 string_add=f"Noise {noise}",
#                 loc="ct",
#                 size=5,
#                 inset_fraction=0.2,
#                 style="b",
#             )

#         # prints the figure
#         if self.Printer is not None and filename is not None:
#             self.Printer.savefig(
#                 fig, filename, label_figs=ax_, size=6, loc="bl", inset_fraction=0.2
#             )

#     @static_dataset_decorator
#     def violin_plot_comparison(self, state, model, X_data, filename):
#         self.set_attributes(**state)

#         df = pd.DataFrame()

#         # uses the model to get the predictions
#         pred_data, scaled_param, params = model.predict(X_data)

#         # scales the parameters
#         scaled_param = self.dataset.SHO_scaler.transform(params)

#         # gets the parameters from the SHO LSQF fit
#         true = self.dataset.SHO_fit_results().reshape(-1, 4)

#         # Builds the dataframe for the violin plot
#         true_df = pd.DataFrame(
#             true, columns=["Amplitude", "Resonance", "Q-Factor", "Phase"]
#         )
#         predicted_df = pd.DataFrame(
#             scaled_param, columns=["Amplitude",
#                                    "Resonance", "Q-Factor", "Phase"]
#         )

#         # merges the two dataframes
#         df = pd.concat((true_df, predicted_df))

#         # adds the labels to the dataframe
#         names = [true, scaled_param]
#         names_str = ["LSQF", "NN"]
#         # ["Amplitude", "Resonance", "Q-Factor", "Phase"]
#         labels = ["A", "\u03C9", "Q", "\u03C6"]

#         # adds the labels to the dataframe
#         for j, name in enumerate(names):
#             for i, label in enumerate(labels):
#                 dict_ = {
#                     "value": name[:, i],
#                     "parameter": np.repeat(label, name.shape[0]),
#                     "dataset": np.repeat(names_str[j], name.shape[0]),
#                 }

#                 df = pd.concat((df, pd.DataFrame(dict_)))

#         # builds the plot
#         fig, ax = plt.subplots(figsize=(2, 2))

#         # plots the data
#         sns.violinplot(
#             data=df, x="parameter", y="value", hue="dataset", split=True, ax=ax
#         )

#         # labels the figure and does some styling
#         labelfigs(ax, 0, style="b")
#         ax.set_ylabel("Scaled SHO Results")
#         ax.set_xlabel("")

#         # Get the legend associated with the plot
#         legend = ax.get_legend()
#         legend.set_title("")

#         # ax.set_aspect(1)

#         # prints the figure
#         if self.Printer is not None and filename is not None:
#             self.Printer.savefig(fig, filename)

#         return fig

#     def violin_plot_comparison_hysteresis(self, model, X_data, filename):

#         df = pd.DataFrame()

#         # uses the model to get the predictions
#         pred_data, scaled_param, params = model.predict(X_data, is_SHO=False)

#         # gets the parameters from the SHO LSQF fit
#         # true = self.dataset.SHO_fit_results().reshape(-1, 4)

#         true = self.dataset.LSQF_hysteresis_params().reshape(-1, 9)

#         true_scaled = self.dataset.loop_param_scaler.transform(true)

#         # Builds the dataframe for the violin plot
#         true_df = pd.DataFrame(
#             true, columns=["a0", "a1", "a2", "a3", "a4",
#                            "b0", "b1", "b2", "b3"]
#         )
#         predicted_df = pd.DataFrame(
#             scaled_param, columns=["a0", "a1", "a2", "a3", "a4",
#                                    "b0", "b1", "b2", "b3"]
#         )

#         # merges the two dataframes
#         df = pd.concat((predicted_df, true_df))

#         # adds the labels to the dataframe
#         names = [true_scaled, scaled_param]
#         names_str = ["NN", "LSQF"]

#         labels = ["a0", "a1", "a2", "a3", "a4", "b0", "b1", "b2", "b3"]

#         # adds the labels to the dataframe
#         for j, name in enumerate(names):
#             for i, label in enumerate(labels):
#                 dict_ = {
#                     "value": name[:, i],
#                     "parameter": np.repeat(label, name.shape[0]),
#                     "dataset": np.repeat(names_str[j], name.shape[0]),
#                 }

#                 df = pd.concat((df, pd.DataFrame(dict_)))

#         # builds the plot
#         fig, ax = plt.subplots(figsize=(4, 4))

#         # plots the data
#         print('df colums',df.columns[df.columns.duplicated()])
#         print('df index',df.index[df.index.duplicated()])

#         # df = df.loc[:, ~df.columns.duplicated()]
#         df = df.reset_index(drop=False)

#         sns.violinplot(
#             data=df, x="parameter", y="value", hue="dataset", split=True, ax=ax, linewidth=.1,
#         )

#         # labels the figure and does some styling
#         labelfigs(ax, 0, style="b")
#         ax.set_ylabel("Scaled SHO Results")
#         ax.set_xlabel("")

#         # Get the legend associated with the plot
#         legend = ax.get_legend()
#         legend.set_title("")

#         # prints the figure
#         if self.Printer is not None and filename is not None:
#             self.Printer.savefig(fig, filename)






#     def MSE_compare(self, true_data, predictions, labels):
#         for pred, label in zip(predictions, labels):
#             if isinstance(pred, nn.Module):
#                 pred_data, scaled_param, parm = pred.predict(true_data)

#             elif isinstance(pred, dict):
#                 pred_data, _ = self.dataset.get_raw_data_from_LSQF_SHO(pred)

#                 pred_data = torch.from_numpy(pred_data)

#             # Computes the MSE
#             out = nn.MSELoss()(true_data, pred_data)

#             # prints the MSE
#             print(f"{label} Mean Squared Error: {out:0.4f}")

#     def get_selected_hysteresis(self,
#                                 data,
#                                 row=None,
#                                 col=None,
#                                 cycle=None):
#         """
#          Function that extracts a dataset or chooses a random dataset from the hysteresis loop

#         Returns:
#             np.array: get selected hysteresis loop
#         """

#         if row is None:
#             row = np.random.randint(0, data.shape[0], 1)

#         if col is None:
#             col = np.random.randint(0, data.shape[1], 1)

#         if cycle is None:
#             cycle = np.random.randint(0, data.shape[2], 1)

#         return (row, col, cycle)

#     def random_hysteresis(self,
#                           raw_hysteresis_loop,
#                           lsqf_hysteresis_loop,
#                           voltage,
#                           filename,
#                           size,
#                           row, col, cycle):


#         fig, ax = subfigures(1, 1, size=size)

#         ax[0].plot(voltage.squeeze(),
#                     raw_hysteresis_loop[row, col, cycle, :].squeeze(), 'o', label="Raw Data")


#         ax[0].plot(voltage.squeeze(),
#                     lsqf_hysteresis_loop[row, col, cycle, :].squeeze(), 'r', label='LSQF')

#         ax[0].set_xlabel('Voltage (V)')
#         ax[0].set_ylabel('Amplitude (Arb. U.)')
#         ax[0].legend()

#         # prints the figure
#         if self.Printer is not None and filename is not None:
#             self.Printer.savefig(fig, filename, label_figs=ax, style="b")


#     def hysteresis_maps(
#         self,
#         parms_pred,
#         colorbars=True,
#         cycle=3,
#         fig_width=10.5,  # figure width in inches
#         filename=None,
#     ):
#         # # reshape data:
#         # if data.shape != 3:

#         # calculates the size of the embedding image
#         embedding_image_size = 60

#         fig, axs = plt.subplots(
#             2,
#             9,
#             figsize=(fig_width, 4),
#             gridspec_kw={"height_ratios": [1, 1]},
#         )

#         parms_lsqf = self.dataset.LSQF_hysteresis_params()[:, :, cycle, :].reshape(-1, 9)
#         parms_pred = parms_pred.reshape(embedding_image_size, embedding_image_size, 4, 9)[:, :, cycle, :].reshape(-1, 9)

#         clims = []

#         colorbar_labels = [
#             'a0', 'a1', 'a2', 'a3', 'a4', 'b0', 'b1', 'b2', 'b3'
#         ]

#         # Titles for each row
#         row_titles = ['Predicted Parameters', 'LSQF Parameters']

#         string_add = 'a'

#         for i in range(9):
#             clims.append(
#                 (
#                     np.min(
#                         [
#                             parms_pred[:, i].min(),
#                             parms_lsqf[:, i].min(),
#                         ]
#                     ),
#                     np.max(
#                         [
#                             parms_pred[:, i].max(),
#                             parms_lsqf[:, i].max(),
#                         ]
#                     ),
#                 )
#             )

#             axs[0, i].imshow(
#                 parms_pred[:, i].reshape(
#                     embedding_image_size, embedding_image_size),
#                 cmap="viridis",
#                 vmin=clims[i][0],
#                 vmax=clims[i][1],
#             )
#             axs[0,i].set_xticklabels('')
#             axs[0,i].set_yticklabels('')
#             axs[1, i].imshow(
#                 parms_lsqf[:, i].reshape(
#                     embedding_image_size, embedding_image_size),
#                 cmap="viridis",
#                 vmin=clims[i][0],
#                 vmax=clims[i][1],
#             )
#             axs[1,i].set_xticklabels('')
#             axs[1,i].set_yticklabels('')

#             if colorbars:
#                 # Create an axis divider for each subplot
#                 divider = make_axes_locatable(axs[1, i])
#                 # Append axes to the bottom of the divider with appropriate padding
#                 cax = divider.append_axes("bottom", size="5%", pad=0.25)
#                 cbar = plt.colorbar(
#                     axs[1, i].images[0], cax=cax, format="%.1e", orientation='horizontal')
#                 # Set the label for each colorbar
#                 cbar.set_label(colorbar_labels[i])

#             labelfigs(axs[0,i],
#                     string_add=colorbar_labels[i],
#                     loc ='ct',
#                     size=8,
#                     inset_fraction=(0.2, 0.2)
#                     )
#              # Update the char to the next order
#             ascii_value = ord(string_add)+1
#             string_add = chr(ascii_value)

#         labelfigs(axs[0,0],
#         string_add='a',
#         loc ='tl',
#         size=8,
#         inset_fraction=(0.2, 0.2)
#         )
#         labelfigs(axs[1,0],
#         string_add='b',
#         loc ='tl',
#         size=8,
#         inset_fraction=(0.2, 0.2)
#         )

#         # Calculate the vertical position for the row titles
#         title_y_positions = [0.85, 0.5]  # You may need to adjust these values

#         # Set the titles for each row using fig.text
#         for i, title in enumerate(row_titles):
#             fig.text(0.5, title_y_positions[i], title, ha='center',
#                      va='center', fontsize=10, transform=fig.transFigure)


#         # prints the figure
#         if self.Printer is not None and filename is not None:
#             print('use printing function')
#             self.Printer.savefig(
#                 fig, filename, size=6, loc="tl", inset_fraction=(0.2, 0.2)
#             )

#     def ranked_mse(self, true, sample_a, other_samples=None):
#         """
#         Compute Mean Squared Error (MSE) between two datasets of samples.

#         Args:
#             true (array-like): First dataset of samples.
#             sample_a (dict): Dictionary with key as sample name and value as dataset of samples.
#             other_samples (dict, optional): Dictionary with key as sample name and value as dataset of samples. Defaults to None.

#         Returns:
#             DataFrame: DataFrame with original index and computed MSE for each sample.
#         """

#         # Extract the key and value
#         sample_a_key, sample_a_value = list(sample_a.items())[0]

#         # Ensure inputs are numpy arrays
#         true = np.array(true)
#         sample_a_value = np.array(sample_a_value)

#         # Calculate MSE for each sample
#         mse = np.mean((true - sample_a_value) ** 2, axis=1)

#         # Create a DataFrame with original index and MSE
#         df = pd.DataFrame({
#             'Original Index': np.arange(len(mse), dtype=int),
#             f'MSE_{sample_a_key}': mse
#         })

#         if other_samples is not None:
#             other_sample_key, other_sample_value = list(other_samples.items())[0]
#             # Calculate MSE for each sample
#             mse_other_sample = np.mean((true - other_sample_value) ** 2, axis=1)
#             # Add the new column to the DataFrame
#             df[f'MSE_{other_sample_key}'] = mse_other_sample

#         # Sort the DataFrame by MSE to find best, worst, and middle examples
#         sorted_df = df.sort_values(f'MSE_{sample_a_key}').reset_index(drop=True)

#         # Identify best, worst, and middle examples and ensure index remains int
#         best_example = sorted_df.iloc[0]
#         best_example['Original Index'] = int(best_example['Original Index'])

#         worst_example = sorted_df.iloc[-1]
#         worst_example['Original Index'] = int(worst_example['Original Index'])

#         middle_example = sorted_df.iloc[len(sorted_df) // 2]
#         middle_example['Original Index'] = int(middle_example['Original Index'])

#         return best_example, middle_example, worst_example

#     def hysteresis_comparison(self,
#                              data,
#                              row=None,
#                              col=None,
#                              cycle=None,
#                              size=(1.25, 1.25),
#                              gaps=(1, 0.66),
#                              nn_model=None,
#                              measurement_state=None,
#                              filename="hysteresis_comparison"):
#         """
#         Plot a comparison of the hysteresis loop.

#         Args:
#             data (list): List of data types to plot.
#             row (int, optional): Row to plot. Defaults to None.
#             col (int, optional): Column to plot. Defaults to None.
#             cycle (int, optional): Cycle to plot. Defaults to None.
#             size (tuple, optional): Size of the image to plot. Defaults to (1.25, 1.25).
#             gaps (tuple, optional): Gaps between subplots. Defaults to (1, 0.66).
#             nn_model (object, optional): Neural network model for comparison. Defaults to None.
#             measurement_state (str, optional): Measurement state to plot. Defaults to None.
#             filename (str, optional): Filename to save the plot. Defaults to "hysteresis_comparison".
#         """

#         # sets the measurement state
#         if self.dataset.measurement_state is not None:
#             self.dataset.measurement_state = measurement_state

#         # if only the LSQF is to be plotted
#         if 'LSQF' in data and 'NN' not in data:
#             # gets the LSQF Hysteresis Loops from the Dataset
#             loops, raw_hysteresis_loop_scaled, voltage = self.dataset.get_LSQF_hysteresis_fits(compare=True, index=False)

#             raw_hysteresis_loop = self.dataset.hysteresis_scaler.inverse_transform(raw_hysteresis_loop_scaled)

#             # selects a point to plot
#             row, col, cycle = self.get_selected_hysteresis(
#                 raw_hysteresis_loop, row, col, cycle)

#             self.random_hysteresis(raw_hysteresis_loop,
#                                    loops,
#                                    voltage,
#                                    filename,
#                                    size,
#                                    row, col, cycle)
#             return

#         # gets the LSQF Hysteresis Loops from the Dataset
#         loops, raw_hysteresis_loop_scaled, voltage = self.dataset.get_LSQF_hysteresis_fits(compare=True)

#         # scales the loops for comparison
#         loops_scaled = self.dataset.hysteresis_scaler.transform(loops)
#         raw_hysteresis_loop = self.dataset.hysteresis_scaler.inverse_transform(raw_hysteresis_loop_scaled)

#         # gets the NN data for comparison
#         if nn_model is not None:
#             # gets the data for model prediction with the NN
#             _data, voltage = self.dataset.get_hysteresis(scaled=True, loop_interpolated=True)
#             _data = torch.atleast_3d(torch.tensor(_data.reshape(-1, self.dataset.voltage_steps_per_cycle))).float()

#             NN_pred_data, NN_scaled_params, NN_params = nn_model.predict(
#                 _data, translate_params=False, is_SHO=False)
#             NN_loops = loop_fitting_function_torch(NN_params, voltage[:, 0].squeeze()).to(
#                 'cpu').detach().numpy().squeeze()
#             NN_loops_scaled = self.dataset.hysteresis_scaler.transform(NN_loops)

#         # if we are plotting the NN and LSQF results
#         fig, ax = subfigures(3, len(data), gaps=gaps, size=size)

#         # loops around the models provided
#         for j, model in enumerate(data):

#             if model == 'LSQF':
#                 out = self.ranked_mse(raw_hysteresis_loop_scaled,
#                                       {'LSQF': loops_scaled},
#                                       {'NN': NN_loops_scaled})

#             elif model == 'NN':
#                 out = self.ranked_mse(raw_hysteresis_loop_scaled,
#                                       {'NN': NN_loops_scaled},
#                                       {'LSQF': loops_scaled})

#             for i, results in enumerate(out):

#                 # sets the index for the plots
#                 plot_idx = i * 2 + j

#                 index = int(results['Original Index'])

#                 ax[plot_idx].plot(voltage,
#                                   raw_hysteresis_loop[index], 'o', label="Raw Data")

#                 ax[plot_idx].plot(voltage,
#                                   loops[index], 'r', label='LSQF')

#                 ax[plot_idx].plot(voltage,
#                                   NN_loops[index], 'g', label='NN')

#                 ax[plot_idx].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

#                 # Position text at (1 inch, 2 inches) from the bottom left corner of the figure
#                 text_position_in_inches = (
#                     -1 * (gaps[0] + size[0]) * ((2 - i) % 3) + size[0] / 2,
#                     (gaps[1] + size[1]) * (1.25 - i // 3 - 1.25) - gaps[1],
#                 )

#                 # gets the axis position in inches - gets the bottom center
#                 center = get_axis_pos_inches(fig, ax[plot_idx])

#                 # selects the text position as an offset from the bottom center
#                 text_position_in_inches = (center[0], center[1] - 0.32 + .125)

#                 error = results['MSE_LSQF']

#                 error_string = f"LSQF MSE: {error:0.4f}"

#                 add_text_to_figure(
#                     fig,
#                     error_string,
#                     text_position_in_inches,
#                     fontsize=6,
#                     ha="center",
#                 )

#                 # selects the text position as an offset from the bottom center
#                 text_position_in_inches = (center[0], center[1] - 0.3)

#                 error = results['MSE_NN']

#                 error_string = f"NN MSE: {error:0.4f}"

#                 add_text_to_figure(
#                     fig,
#                     error_string,
#                     text_position_in_inches,
#                     fontsize=6,
#                     ha="center",
#                 )

#                 ax[plot_idx - 1].set_ylabel("(Arb. U.)")
#                 ax[plot_idx].set_ylabel("(Arb. U.)")

#         # add a legend just for the last one
#         lines, labels = ax[plot_idx - 1].get_legend_handles_labels()
#         ax[plot_idx - 1].legend(lines, labels, loc="upper right")
#         lines, labels = ax[plot_idx].get_legend_handles_labels()
#         ax[plot_idx].legend(lines, labels, loc="upper right")

#         ax[plot_idx - 1].set_xlabel("Voltage (V)")
#         ax[plot_idx].set_xlabel("Voltage (V)")
#         ax[plot_idx - 1].xaxis.set_label_coords(0.5, -0.28)
#         ax[plot_idx].xaxis.set_label_coords(0.5, -0.28)

#         # prints the figure
#         if self.Printer is not None and filename is not None:
#             self.Printer.savefig(fig, filename, label_figs=ax, style="b")
