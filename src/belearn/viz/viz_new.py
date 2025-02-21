from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Type

from belearn.dataset.dataset_new import BE_Dataset
from belearn.dataset.State import State

import numpy as np

from m3util.viz.layout import (
    layout_fig,
    # add_box,
    # inset_connector,
    # scalebar,
    # imagemap,
    # FigDimConverter,
    # subfigures,
    # get_axis_pos_inches,
    # draw_line_with_text,
)

from m3util.viz.arrows import (
    draw_ellipse_with_arrow,
    #DrawArrow,
   # draw_extended_arrow_indicator,
)

# functions, attributes, and methods in Viz class: 
# plot_magnitude_spectrum


# Defines the color palettes for the plots
color_palette = {
    "LSQF_A": "#003f5c",  # dark blue
    "LSQF_P": "#444e86",  # bluish purple
    "NN_A": "#955196",  # purple
    "NN_P": "#dd5182",  # pinkish red
    "real": "#ff6e54",  # orange
    "imag": "#ffa600",  # yellow-orange
    "mag": "#2f9eaa",  # cyan
    "phase": "#66c21f",  # green
}

@dataclass
class Viz(State):
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
    
    # SHO_labels: List[Dict[str, str]] = field(
    #     default_factory=lambda: [
    #         {"title": "Amplitude", "y_label": "Amplitude \n (Arb. U.)"},
    #         {"title": "Resonance Frequency", "y_label": "Resonance Frequency \n (Hz)"},
    #         {"title": "Dampening", "y_label": "Quality Factor \n (Arb. U.)"},
    #         {"title": "Phase", "y_label": "Phase \n (rad)"},
    #     ]
    # )

    
    def __init__(self, 
                 dataset: Any, 
                 Printer: Optional[Type] = None, 
                 verbose: bool = False, 
                 labelfigs_: bool = True, 
                 image_scalebar: Optional[Any] = None,
                 color_palette: Optional[Any] = None,
                 SHO_ranges: Optional[Any] = None,
                 SHO_labels: Optional[List[Dict[str, str]]]= field(
                                    default_factory=lambda: [
                                        {"title": "Amplitude", "y_label": "Amplitude \n (Arb. U.)"},
                                        {"title": "Resonance Frequency", "y_label": "Resonance Frequency \n (Hz)"},
                                        {"title": "Dampening", "y_label": "Quality Factor \n (Arb. U.)"},
                                        {"title": "Phase", "y_label": "Phase \n (rad)"},
                                    ]
                                )
                 ):
        super().__init__()
        self.dataset = dataset
        self.Printer = Printer
        self.verbose = verbose
        self.labelfigs_ = labelfigs_
        self.image_scalebar = image_scalebar
        self.color_palette = color_palette
        self.SHO_ranges = SHO_ranges
        if SHO_labels is not None:
            self.SHO_labels = SHO_labels
    
    # dataset: Any  # Specify the type based on what you expect
    # # You can also define the type of Printer if you know it
    # Printer: Optional[Type] = None
    # verbose: bool = False
    # labelfigs_: bool = True
    # # Specify the type based on what you expect
    # SHO_ranges: Optional[Any] = None
    # # Specify the type based on what you expect
    # image_scalebar: Optional[Any] = None

    # SHO_labels: List[Dict[str, str]] = field(
    #     default_factory=lambda: [
    #         {"title": "Amplitude", "y_label": "Amplitude \n (Arb. U.)"},
    #         {"title": "Resonance Frequency", "y_label": "Resonance Frequency \n (Hz)"},
    #         {"title": "Dampening", "y_label": "Quality Factor \n (Arb. U.)"},
    #         {"title": "Phase", "y_label": "Phase \n (rad)"},
    #     ]
    # )

    # # Replace Any with the expected type if known
    # color_palette: Optional[Any] = None
    
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
                self.measurement_state == "on"
                or self.measurement_state == "off"
            ):
                voltage_step = np.random.randint(0, self.dataset.voltage_steps // 2)
            else:
                # Otherwise, select from the full range of voltage steps
                voltage_step = np.random.randint(0, self.dataset.voltage_steps)

        # Return the determined or provided voltage step index
        return voltage_step

    
    ##### Methods #####
    
    @static_dataset_decorator
    def plot_magnitude_spectrum(
        self,
        ax1,
        true,
        predict=None,
        pixel=None,
        voltage_step=None,
        fig=None,
        add_arrows=None,
        add_labels=False,
        annotation_kwargs={},
        line_kwargs={},
        arrowprops={},
        halo={},
        **kwargs,
    ):
        # Set the attributes for the true dataset
        self.set_attributes(**true)

        # If a pixel is not provided, select a random pixel
        if pixel is None:
            pixel = np.random.randint(0, self.dataset.num_pix)

        # Get the voltage step, considering the current state
        voltage_step = self.get_voltage_step(voltage_step)

        # Set dataset state to grab the magnitude spectrum
        self.raw_format = "magnitude spectrum"

        # Get the raw spectral data for the selected pixel and voltage step
        data, x = self.raw_spectra(pixel, voltage_step, frequency=True)

        # Plot amplitude and phase for the true dataset
        ax1.plot(
            x,
            data[0].flatten(),
            color=color_palette["mag"],
            marker="s",
            label=self.dataset.label + " Amplitude",
        )
        ax2 = ax1.twinx()
        ax2.plot(
            x,
            data[1].flatten(),
            color=color_palette["phase"],
            marker="s",
            label=self.dataset.label + " Phase",
        )

        # Ensure ax2 is drawn on top of ax1 by setting a higher zorder
        ax1.set_zorder(ax2.get_zorder() + 1)

        # Remove the axes background (set to transparent)
        ax1.set_facecolor("none")

        # If a predicted dataset is provided, plot its amplitude and phase
        if predict is not None:
            self.set_attributes(**predict)
            data, x = self.raw_spectra(
                pixel, voltage_step, frequency=True, **kwargs
            )
            ax1.plot(
                x, data[0].flatten(), "bo", label=self.dataset.label + " Amplitude"
            )
            ax2.plot(x, data[1].flatten(), "ro", label=self.dataset.label + " Phase")
            self.set_attributes(**true)

        # Label the axes for the first subplot
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Amplitude (Arb. U.)")
        ax2.set_ylabel("Phase (rad)")

        self._scientific_notation_dual(ax1, ax2)

        if add_arrows is not None:
            # Mandatory keys that must be present
            required_keys = ["mag_value", "phase_value", "width", "height"]

            # Check if required keys are present
            missing_keys = [key for key in required_keys if key not in add_arrows]
            if missing_keys:
                raise ValueError(
                    f"Missing required parameters in add_arrows: {', '.join(missing_keys)}"
                )

            draw_ellipse_with_arrow(
                ax1,
                x,
                data[0].flatten(),
                add_arrows["mag_value"],
                add_arrows["width"],
                add_arrows["height"],
                axis=add_arrows.get("axis", "x"),
                line_direction=add_arrows.get("line_direction", "horizontal"),
                arrow_position=add_arrows.get("arrow_position", "top"),
                arrow_length_frac=add_arrows.get("arrow_length_frac", 0.2),
                color=add_arrows.get("color", color_palette["mag"]),
                linewidth=add_arrows.get("linewidth", 1),
                arrow_props=add_arrows.get(
                    "arrow_props",
                    {
                        "facecolor": color_palette["mag"],
                        "width": 2,
                        "headwidth": 8,  # Arrowhead width in points
                        "headlength": 10,  # Arrowhead length in points
                        "linewidth": 0,
                    },
                ),
                ellipse_props=add_arrows.get("ellipse_props", None),
                arrow_direction="negative",
            )

            draw_ellipse_with_arrow(
                ax2,
                x,
                data[1].flatten(),
                add_arrows["phase_value"],
                add_arrows["width"],
                add_arrows["height"],
                axis=add_arrows.get("axis", "x"),
                line_direction=add_arrows.get("line_direction", "horizontal"),
                arrow_position=add_arrows.get("arrow_position", "bottom"),
                arrow_length_frac=add_arrows.get("arrow_length_frac", 0.2),
                color=add_arrows.get("color", color_palette["phase"]),
                linewidth=add_arrows.get("linewidth", 1),
                arrow_props=add_arrows.get(
                    "arrow_props",
                    {
                        "facecolor": color_palette["phase"],
                        "width": 2,
                        "headwidth": 8,  # Arrowhead width in points
                        "headlength": 10,  # Arrowhead length in points
                        "linewidth": 0,
                    },
                ),
                ellipse_props=add_arrows.get("ellipse_props", None),
                arrow_direction="positive",
            )

        if add_labels:
            _ax = (ax1, ax2)
            self._SHO_labels(
                fig,
                ax=_ax,
                x=x,
                data=data,
                line_kwargs=line_kwargs,
                annotation_kwargs=annotation_kwargs,
                arrowprops=arrowprops,
                halo=halo,
            )

        return ax1, ax2
    
    @static_dataset_decorator
    def plot_real_imaginary(
        self,
        ax1,
        true,
        predict=None,
        pixel=None,
        voltage_step=None,
        add_arrows=None,
        **kwargs,
    ):
        # Set the attributes for the true dataset
        self.set_attributes(**true)

        # Reset dataset state to complex format
        self.dataset.raw_format = "complex"

        # Get the complex raw spectral data for the selected pixel and voltage step
        data, x = self.raw_spectra(pixel, voltage_step, frequency=True)

        # Plot real and imaginary components for the true dataset
        ax1.plot(
            x,
            data[0].flatten(),
            color=color_palette["real"],
            marker="o",
            markeredgecolor="k",
            markeredgewidth=0.02,
            markersize=1,
            label=self.dataset.label + " Real",
        )
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Real (Arb. U.)")
        ax2 = ax1.twinx()
        ax2.set_ylabel("Imag (Arb. U.)")
        ax2.plot(
            x,
            data[1].flatten(),
            color=color_palette["imag"],
            marker="s",
            markeredgecolor="k",
            markeredgewidth=0.02,
            markersize=1,
            label=self.dataset.label + " Imag",
        )

        # If a predicted dataset is provided, plot its real and imaginary components
        if predict is not None:
            self.set_attributes(**predict)
            data, x = self.raw_spectra(
                pixel, voltage_step, frequency=True, **kwargs
            )
            ax1.plot(x, data[0].flatten(), "ko", label=self.dataset.label + " Real")
            ax2.plot(x, data[1].flatten(), "gs", label=self.dataset.label + " Imag")
            self.set_attributes(**true)

        self._scientific_notation_dual(ax1, ax2)

        if add_arrows is not None:
            # Mandatory keys that must be present
            required_keys = ["imag_value", "real_value", "width", "height"]

            # Check if required keys are present
            missing_keys = [key for key in required_keys if key not in add_arrows]
            if missing_keys:
                raise ValueError(
                    f"Missing required parameters in add_arrows: {', '.join(missing_keys)}"
                )

            draw_ellipse_with_arrow(
                ax1,
                x,
                data[0].flatten(),
                add_arrows["real_value"],
                add_arrows["width"],
                add_arrows["height"],
                axis=add_arrows.get("axis", "x"),
                line_direction=add_arrows.get("line_direction", "horizontal"),
                arrow_position=add_arrows.get("arrow_position", "top"),
                arrow_length_frac=add_arrows.get("arrow_length_frac", 0.2),
                color=add_arrows.get("color", color_palette["real"]),
                linewidth=add_arrows.get("linewidth", 1),
                arrow_props=add_arrows.get(
                    "arrow_props",
                    {
                        "facecolor": color_palette["real"],
                        "width": 2,
                        "headwidth": 8,  # Arrowhead width in points
                        "headlength": 10,  # Arrowhead length in points
                        "linewidth": 0,
                    },
                ),
                ellipse_props=add_arrows.get("ellipse_props", None),
                arrow_direction="negative",
            )

            draw_ellipse_with_arrow(
                ax2,
                x,
                data[1].flatten(),
                add_arrows["imag_value"],
                add_arrows["width"],
                add_arrows["height"],
                axis=add_arrows.get("axis", "x"),
                line_direction=add_arrows.get("line_direction", "horizontal"),
                arrow_position=add_arrows.get("arrow_position", "bottom"),
                arrow_length_frac=add_arrows.get("arrow_length_frac", 0.2),
                color=add_arrows.get("color", color_palette["imag"]),
                linewidth=add_arrows.get("linewidth", 1),
                arrow_props=add_arrows.get(
                    "arrow_props",
                    {
                        "facecolor": color_palette["imag"],
                        "width": 2,
                        "headwidth": 8,  # Arrowhead width in points
                        "headlength": 10,  # Arrowhead length in points
                        "linewidth": 0,
                    },
                ),
                ellipse_props=add_arrows.get("ellipse_props", None),
                arrow_direction="positive",
            )

        return ax1, ax2
    
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
            true (dict): Attributes of the true dataset to be set.
            predict (dict, optional): Attributes of the predicted dataset to be set. Defaults to None.
            filename (str, optional): Name of the file to save the figure. Defaults to None.
            pixel (int, optional): Pixel index to plot. If None, a random pixel is selected. Defaults to None.
            voltage_step (int, optional): Voltage step index to plot. If None, it is determined by the dataset. Defaults to None.
            legend (bool, optional): Whether to display a legend on the plot. Defaults to True.
            **kwargs: Additional keyword arguments for the dataset's raw_spectra method.

        Returns:
            None
        """

        # Set the attributes for the true dataset
        self.set_attributes(**true)

        # Initialize figure and axes for plotting
        fig, axs = layout_fig(2, 2, figsize=(5, 1.25))

        ax_mag, ax_phase = self.plot_magnitude_spectrum(
            axs[0], true, predict, pixel, voltage_step, fig=fig, **kwargs
        )

        ax_real, ax_imag = self.plot_real_imaginary(
            axs[1], true, predict, pixel, voltage_step, **kwargs
        )

        # Adjust the format of the tick labels and box aspect for all axes
        axes = [ax_mag, ax_real, ax_phase, ax_imag]

        for ax in axes:
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
            self.Printer.savefig(
                fig, filename, label_figs=[ax_phase, ax_imag], style="bw", loc="bl"
            )
