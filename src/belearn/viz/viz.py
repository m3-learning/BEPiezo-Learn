import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Type, Callable

from belearn.dataset.model_utils import BE_model_utils
from belearn.util.wrappers import context_manager_decorator
from belearn.dataset.analytics import MSE
from belearn.functions.sho import SHO_nn
#from belearn.dataset.transformers import to_real_imag, to_complex

from autophyslearn.spectroscopic.nn import Multiscale1DFitter, Model
from autophyslearn.postprocessing.complex import ComplexPostProcessor

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec

import pandas as pd
import seaborn as sns

from scipy import fftpack

import torch
from torch import nn


from contextlib import contextmanager
import inspect

from m3util.ml.rand import set_seeds

from m3util.viz.layout import (
    layout_fig,
    add_box,
    inset_connector,
    scalebar,
    imagemap,
    FigDimConverter,
    subfigures,
    get_axis_pos_inches,
    # draw_line_with_text,
)

from m3util.viz.arrows import (
    draw_ellipse_with_arrow,
    # DrawArrow,
    # draw_extended_arrow_indicator,
)

from m3util.viz.text import (
    add_text_to_figure,
    set_sci_notation_label,
    labelfigs,
    number_to_letters,
    # obj_offset,
)


from m3util.util.IO import make_folder
from m3util.viz.movies import make_movie

# functions, attributes, and methods in Viz class:
# plot_magnitude_spectrum


# Defines the color palettes for the plots
color_palette = {
    "LSQF_A": "#003f5c",  # dark blue
    "LSQF_P": "#444e86",  # bluish purple
    "NN_A": "#955196",  # purple
    "NN_P": "#dd5182",  # pinkish red
    "mag": "#2f9eaa",  # cyan
    "phase": "#66c21f",  # green
    "real": "#ff6e54",  # orange
    "imag": "#ffa600",  # yellow-orange
    "true_mag": "#2f9eaa",  # cyan
    "true_phase": "#66c21f",  # green
    "true_real": "#ff6e54",  # orange
    "true_imag": "#ffa600",  # yellow-orange
    "predict_mag": "#1A237E",  # dark blue
    "predict_phase": "#003300",  # dark green
    "predict_real": "#D84315",  # dark orange
    "predict_imag": "#b37400",  # dark yellowish tan
}


class Viz(BE_model_utils):
    """
    A DataClass for handling various visualization settings and data.

    Attributes:
        dataset (Any): The dataset to visualize. Replace `Any` with the specific type.
        printer (Optional[Type], optional): printer for output. Defaults to None.
        verbose (bool, optional): Verbosity flag. Defaults to False.
        labelfigs_ (bool, optional): Flag to label figures. Defaults to True.
        SHO_ranges (Optional[Any], optional): Ranges for SHO data. Defaults to None.
        image_scalebar (Optional[Any], optional): Scalebar settings for images. Defaults to None.
        SHO_labels (List[Dict[str, str]], optional): Labels for SHO data. Defaults to predefined list.
        color_palette (Optional[Any], optional): Color palette settings. Defaults to None.
        hysteresis_function (Callable, optional): The hysteresis function for processing. Defaults to hysteresis_nn.


    """

    # SHO_labels: List[Dict[str, str]] = field(
    #     default_factory=lambda: [
    #         {"title": "Amplitude", "y_label": "Amplitude \n (Arb. U.)"},
    #         {"title": "Resonance Frequency", "y_label": "Resonance Frequency \n (Hz)"},
    #         {"title": "Dampening", "y_label": "Quality Factor \n (Arb. U.)"},
    #         {"title": "Phase", "y_label": "Phase \n (rad)"},
    #     ]
    # )

    def __init__(
        self,
        dataset: Any,
        printer: Optional[Type] = None,
        verbose: bool = False,
        labelfigs_: bool = True,
        image_scalebar: Optional[Any] = None,
        color_palette: Optional[Any] = None,
        SHO_ranges: Optional[Any] = None,
        SHO_labels: Optional[List[Dict[str, str]]] = None,

    ):
        self.dataset = dataset
        self.printer = printer
        self.verbose = verbose
        self.labelfigs_ = labelfigs_
        self.image_scalebar = image_scalebar
        self.color_palette = color_palette
        self.SHO_ranges = SHO_ranges
        self.SHO_labels = (
            SHO_labels
            if SHO_labels is not None
            else [
                {"title": "Amplitude", "y_label": "Amplitude \n (Arb. U.)"},
                {
                    "title": "Resonance Frequency",
                    "y_label": "Resonant \n Frequency \n (Hz)",
                },
                {"title": "Dampening", "y_label": "Quality Factor \n (Arb. U.)"},
                {"title": "Phase", "y_label": "Phase \n (rad)"},
            ]
        )

        super().__init__()

    # dataset: Any  # Specify the type based on what you expect
    # # You can also define the type of printer if you know it
    # printer: Optional[Type] = None
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

    ###### SETTERS ######

    # def set_attributes(self, **kwargs):
    #     """
    #     Sets the attributes of the dataset using key-value pairs from a dictionary.

    #     This utility function iterates over the provided keyword arguments and sets
    #     the corresponding attributes of the dataset object. It also ensures that any
    #     necessary setters are triggered, such as for the 'noise' attribute.

    #     Args:
    #         **kwargs:
    #             Arbitrary keyword arguments representing the attributes to set on the dataset.
    #             The keys represent attribute names, and the values represent the values to be set.

    #     Returns:
    #         None
    #     """

    #     # # Iterate over the key-value pairs in kwargs and set the corresponding attributes on the dataset
    #     # for key, value in kwargs.items():
    #     #     setattr(self, key, value)

    #     # # Ensure that the setter for 'noise' is called if the 'noise' attribute is provided in kwargs
    #     # if kwargs.get("noise"):
    #     #     self.noise = kwargs.get("noise")

    #     self.__dict__.update(kwargs)

    # @contextmanager
    # def temporary_state(obj, **modifications):
    #     # Create a deep copy of the object's state
    #     original_state = obj.get_state
    #     try:
    #         # Apply modifications to the object
    #         obj.set_attributes(**modifications)
    #         yield obj
    #     finally:
    #         # Restore the original state
    #         obj.set_attributes(**original_state)

    ##### Methods #####

    # @State.static_dataset_decorator
    @context_manager_decorator
    def plot_twin_axis(
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

        # If a pixel is not provided, select a random pixel
        if pixel is None:
            pixel = np.random.randint(0, self.num_pix)
            self.pixel = pixel  # JGoddy: is this necessary?

        if voltage_step is None:
            # Get the voltage step, considering the current state
            voltage_step = self.get_voltage_step(voltage_step)
            self.voltage_step = voltage_step  # JGoddy: is this necessary

        if "raw_format" in kwargs.keys():
            self.raw_format = kwargs["raw_format"]

        # Get the raw spectral data for the selected pixel and voltage step
        # with State.temporary_state(self, **true):
        # JGoddy added scaled on May 14 without testing it
        # if 'scaled' should always be False here, then remove it from the function call
        data, x = self.raw_spectra(pixel, voltage_step, frequency=True, scaled = self.scaled) 

        # Get the valid parameters for the plot method
        plot_params = mlines.Line2D([], []).properties().keys()

        # Extract kwargs for ax1 and ax2 based on valid plot parameters

        # Remove the prefixes for ax1 and ax2 kwargs
        ax1_kwargs = {
            k[len("ax1_") :]: v
            for k, v in kwargs.items()
            if k[len("ax1_") :] in plot_params and k.startswith("ax1_")
        }
        ax1_true_kwargs = {
            k[len("ax1_true_") :]: v
            for k, v in kwargs.items()
            if k[len("ax1_true_") :] in plot_params and k.startswith("ax1_true_")
        }
        ax1_predict_kwargs = {
            k[len("ax1_predict_") :]: v
            for k, v in kwargs.items()
            if k[len("ax1_predict_") :] in plot_params and k.startswith("ax1_predict_")
        }
        ax2_kwargs = {
            k[len("ax2_") :]: v
            for k, v in kwargs.items()
            if k[len("ax2_") :] in plot_params and k.startswith("ax2_")
        }
        ax2_true_kwargs = {
            k[len("ax2_true_") :]: v
            for k, v in kwargs.items()
            if k[len("ax2_true_") :] in plot_params and k.startswith("ax2_true_")
        }
        ax2_predict_kwargs = {
            k[len("ax2_predict_") :]: v
            for k, v in kwargs.items()
            if k[len("ax2_predict_") :] in plot_params and k.startswith("ax2_predict_")
        }
        # Extract kwargs for either plot
        either_axis_kwargs_for_plotting = {
            k: v
            for k, v in kwargs.items()
            if k in plot_params
            and not k.startswith("ax1_")
            and not k.startswith("ax2_")
        }

        ax1.plot(
            x,
            data[0].flatten(),
            # color = kwargs["ax1_color"],
            # marker = kwargs["marker"],
            # label = kwargs["ax1_label"],
            **ax1_kwargs,
            **ax1_true_kwargs,
            **either_axis_kwargs_for_plotting,
        )

        ax2 = ax1.twinx()
        ax2.plot(
            x,
            data[1].flatten(),
            # color = kwargs["color"][1],
            # marker = kwargs["marker"],
            # label = kwargs["label"][1],
            **ax2_kwargs,
            **ax2_true_kwargs,
            **either_axis_kwargs_for_plotting,
        )

        # Ensure ax2 is drawn on top of ax1 by setting a higher zorder
        # ax1.set_zorder(ax2.get_zorder() + 1)

        # Remove the axes background (set to transparent)
        ax1.set_facecolor("none")

        # If a predicted dataset is provided, plot its:
        # (amplitude and phase) or (real and imaginary components) etc.
        if predict is not None:
            self.set_attributes(**predict)
            # JGoddy added scaled on May 14 without testing it
            # if 'scaled' should always be False here, then remove it from the function call
            data_predict, x = self.raw_spectra(
                pixel, voltage_step, frequency=True, scaled = self.scaled, **kwargs
            )
            ax1.plot(
                x,
                data_predict[0].flatten(),
                color=ax1_predict_kwargs["color"],
                marker="o",
                linestyle=(5, (10, 3)),
                label=ax1_predict_kwargs[
                    "label"
                ],  # self.label + " " + ax1_kwargs["label"]
                # **ax1_predict_kwargs
            )
            ax2.plot(
                x,
                data_predict[1].flatten(),
                color=ax2_predict_kwargs["color"],
                marker="o",
                linestyle=(5, (10, 3)),
                label=ax2_predict_kwargs[
                    "label"
                ],  # self.label + " " + ax2_kwargs["label"]
                # **ax2_predict_kwargs
            )
            self.set_attributes(**true)

        ax1.set_xlabel(kwargs.get("x_label"))
        ax1.set_ylabel(kwargs.get("y1_label"))
        ax2.set_ylabel(kwargs.get("y2_label"))

        self._scientific_notation_dual(ax1, ax2)

        # Add the legend

        # Add the arrows

        return ax1, ax2

    # @State.static_dataset_decorator
    @context_manager_decorator
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
            pixel = np.random.randint(0, self.num_pix)
            self.pixel = pixel

        # Get the voltage step, considering the current state
        voltage_step = self.get_voltage_step(voltage_step)
        self.voltage_step = voltage_step
        # Set dataset state to grab the magnitude spectrum
        self.raw_format = "magnitude spectrum"

        # Get the raw spectral data for the selected pixel and voltage step
        # JGoddy added scaled on May 14 without testing it
        # if 'scaled' should always be False here, then remove it from the function call
        data, x = self.raw_spectra(pixel, voltage_step, frequency=True, scaled = self.scaled)

        # Plot amplitude and phase for the true dataset
        ax1.plot(
            x,
            data[0].flatten(),
            color=color_palette["mag"],
            marker="s",
            label="True " + self.label + " Amplitude",
        )
        ax2 = ax1.twinx()
        ax2.plot(
            x,
            data[1].flatten(),
            color=color_palette["phase"],
            marker="s",
            label="True " + self.label + " Phase",
        )

        # Ensure ax2 is drawn on top of ax1 by setting a higher zorder
        ax1.set_zorder(ax2.get_zorder() + 1)

        # Remove the axes background (set to transparent)
        ax1.set_facecolor("none")

        # If a predicted dataset is provided, plot its amplitude and phase
        if predict is not None:
            self.set_attributes(**predict)
            # JGoddy added scaled on May 14 without testing it
            # if 'scaled' should always be False here, then remove it from the function call
            data, x = self.raw_spectra(pixel, voltage_step, frequency=True, scaled = self.scaled, **kwargs)
            ax1.plot(x, data[0].flatten(), "bo", label=self.label + " Amplitude")
            ax2.plot(x, data[1].flatten(), "ro", label=self.label + " Phase")
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
                ax1,  # ax
                x,  # x_data
                data[0].flatten(),  # y_data
                add_arrows["mag_value"],  # value
                add_arrows["width"],  # width
                add_arrows["height"],  # height
                axis=add_arrows.get("axis", "x"),  # axis
                line_direction=add_arrows.get(
                    "line_direction", "horizontal"
                ),  # line_direction
                arrow_position=add_arrows.get(
                    "arrow_position", "top"
                ),  # arrow_position
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

    # @State.static_dataset_decorator
    @context_manager_decorator
    def plot_real_imaginary(
        self,
        ax1,
        true,
        predict=None,
        pixel=130,  # None
        voltage_step=149,  # None,
        add_arrows=None,
        **kwargs,
    ):
        # Set the attributes for the true dataset
        self.set_attributes(**true)

        # Reset dataset state to complex format
        self.raw_format = "complex"

        # Get the complex raw spectral data for the selected pixel and voltage step
        # JGoddy added scaled on May 14 without testing it
        # if 'scaled' should always be False here, then remove it from the function call
        data, x = self.raw_spectra(pixel, voltage_step, frequency=True, scaled = self.scaled)

        # Plot real and imaginary components for the true dataset
        ax1.plot(
            x,
            data[0].flatten(),
            color=color_palette["real"],
            marker="o",
            markeredgecolor="k",
            markeredgewidth=0.02,
            markersize=1,
            label=self.label + " Real",
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
            label=self.label + " Imag",
        )

        # If a predicted dataset is provided, plot its real and imaginary components
        if predict is not None:
            self.set_attributes(**predict)
            # JGoddy added scaled on May 14 without testing it
            # if 'scaled' should always be False here, then remove it from the function call
            data, x = self.raw_spectra(pixel, voltage_step, frequency=True, scaled = self.scaled, **kwargs)
            ax1.plot(x, data[0].flatten(), "ko", label=self.label + " Real")
            ax2.plot(x, data[1].flatten(), "gs", label=self.label + " Imag")
            self.set_attributes(**true)

        self._scientific_notation_dual(ax1, ax2)

        # if add_arrows is not None:
        #     # Mandatory keys that must be present
        #     required_keys = ["imag_value", "real_value", "width", "height"]

        #     # Check if required keys are present
        #     missing_keys = [key for key in required_keys if key not in add_arrows]
        #     if missing_keys:
        #         raise ValueError(
        #             f"Missing required parameters in add_arrows: {', '.join(missing_keys)}"
        #         )

        #     draw_ellipse_with_arrow(
        #         ax1,
        #         x,
        #         data[0].flatten(),
        #         add_arrows["real_value"],
        #         add_arrows["width"],
        #         add_arrows["height"],
        #         axis=add_arrows.get("axis", "x"),
        #         line_direction=add_arrows.get("line_direction", "horizontal"),
        #         arrow_position=add_arrows.get("arrow_position", "top"),
        #         arrow_length_frac=add_arrows.get("arrow_length_frac", 0.2),
        #         color=add_arrows.get("color", color_palette["real"]),
        #         linewidth=add_arrows.get("linewidth", 1),
        #         arrow_props=add_arrows.get(
        #             "arrow_props",
        #             {
        #                 "facecolor": color_palette["real"],
        #                 "width": 2,
        #                 "headwidth": 8,  # Arrowhead width in points
        #                 "headlength": 10,  # Arrowhead length in points
        #                 "linewidth": 0,
        #             },
        #         ),
        #         ellipse_props=add_arrows.get("ellipse_props", None),
        #         arrow_direction="negative",
        #     )

        #     draw_ellipse_with_arrow(
        #         ax2,
        #         x,
        #         data[1].flatten(),
        #         add_arrows["imag_value"],
        #         add_arrows["width"],
        #         add_arrows["height"],
        #         axis=add_arrows.get("axis", "x"),
        #         line_direction=add_arrows.get("line_direction", "horizontal"),
        #         arrow_position=add_arrows.get("arrow_position", "bottom"),
        #         arrow_length_frac=add_arrows.get("arrow_length_frac", 0.2),
        #         color=add_arrows.get("color", color_palette["imag"]),
        #         linewidth=add_arrows.get("linewidth", 1),
        #         arrow_props=add_arrows.get(
        #             "arrow_props",
        #             {
        #                 "facecolor": color_palette["imag"],
        #                 "width": 2,
        #                 "headwidth": 8,  # Arrowhead width in points
        #                 "headlength": 10,  # Arrowhead length in points
        #                 "linewidth": 0,
        #             },
        #         ),
        #         ellipse_props=add_arrows.get("ellipse_props", None),
        #         arrow_direction="positive",
        #     )

        return ax1, ax2

    # @State.static_dataset_decorator
    @context_manager_decorator
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
        fig, axs = layout_fig(2, 2, figsize=(4.75, 1.25))

        if pixel is None:
            pixel = 130
        if voltage_step is None:
            voltage_step = 149

        print("***")
        print("pixel: ", pixel)
        print("voltage_step: ", voltage_step)
        print("***")

        ax_mag, ax_phase = self.plot_twin_axis(
            axs[0],
            true,
            predict,
            pixel,
            voltage_step,
            fig=fig,
            raw_format="magnitude spectrum",
            ax1_true_color=color_palette["true_mag"],
            ax2_true_color=color_palette["true_phase"],
            ax1_predict_color=color_palette["predict_mag"],
            ax2_predict_color=color_palette["predict_phase"],
            ax1_predict_marker="o",
            ax2_predict_marker="o",
            marker="s",
            ax1_true_label=true["label"] + " Amplitude",
            ax2_true_label=true["label"] + " Phase",
            ax1_predict_label=predict["label"] + " Amplitude"
            if predict is not None
            else None,
            ax2_predict_label=predict["label"] + " Phase"
            if predict is not None
            else None,
            x_label="Frequency (Hz)",
            y1_label="Amplitude (Arb. U.)",
            y2_label="Phase (deg)",
            **kwargs,
        )

        ax_real, ax_imag = self.plot_twin_axis(
            axs[1],
            true,
            predict,
            pixel=pixel,
            voltage_step=voltage_step,
            fig=fig,
            raw_format="complex",
            ax1_true_color=color_palette["true_real"],
            ax2_true_color=color_palette["true_imag"],
            ax1_predict_color=color_palette["predict_real"],
            ax2_predict_color=color_palette["predict_imag"],
            ax1_predict_marker="o",
            ax2_predict_marker="o",
            marker="s",
            ax1_true_label=true["label"] + " Real",
            ax2_true_label=true["label"] + " Imag",
            ax1_predict_label=predict["label"] + " Real"
            if predict is not None
            else None,
            ax2_predict_label=predict["label"] + " Imag"
            if predict is not None
            else None,
            y1_label="Real (Arb. U.)",
            y2_label="Imag (Arb. U.)",
            x_label="Frequency (Hz)",
            **kwargs,
        )

        # ax_mag, ax_phase = self.plot_magnitude_spectrum(
        #     axs[0], true, predict, pixel, voltage_step, fig=fig, **kwargs
        # )

        # ax_real, ax_imag = self.plot_real_imaginary(
        #     axs[1], true, predict, pixel, voltage_step, **kwargs
        # )

        # Adjust the format of the tick labels and box aspect for all axes
        # axes = [ax_mag, ax_real, ax_phase, ax_imag]

        axes = [ax_mag, ax_phase, ax_real, ax_imag]

        for ax in axes:
            ax.set_box_aspect(1)

        # Optionally print the dataset states
        if self.verbose:
            print("True \n")
            self.set_attributes(**true)
            self.extraction_state
            if predict is not None:
                print("predicted \n")
                self.set_attributes(**predict)
                self.extraction_state

        # Display the legend if requested
        if legend:
            # for now, hard code if len(ax_mag.lines) == 2 or 1? what if its more than 2? should I loop over however
            # many lines there are? what is the best way to do this?
            try:
                handles = [
                    ax_mag.lines[0],
                    ax_mag.lines[1],
                    ax_phase.lines[0],
                    ax_phase.lines[1],
                    ax_real.lines[0],
                    ax_real.lines[1],
                    ax_imag.lines[0],
                    ax_imag.lines[1],
                ]
            except:
                handles = [
                    ax_mag.lines[0],
                    ax_phase.lines[0],
                    ax_real.lines[0],
                    ax_imag.lines[0],
                ]

            labels = [h.get_label() for h in handles]
            fig.legend(
                handles,
                labels,
                bbox_to_anchor=(1.0, 1),
                loc="upper right",
                borderaxespad=0.1,
            )

        # Save the figure if a printer object and filename are provided
        if self.printer is not None and filename is not None:
            self.printer.savefig(
                fig, filename, label_figs=[ax_phase, ax_imag], style="bw", loc="bl"
            )

    def _scientific_notation_dual(self, ax1, ax2):
        set_sci_notation_label(
            ax1,
            corner="top left",
            axis="y",
            stroke_color="w",
            linewidth=0.5,
            write_to_axis=ax2,
        )
        set_sci_notation_label(
            ax2, corner="top right", axis="y", stroke_color="w", linewidth=0.5
        )
        set_sci_notation_label(
            ax1, axis="x", stroke_color="w", linewidth=0.5, write_to_axis=ax2
        )
        set_sci_notation_label(ax2, axis="x", stroke_color="w", linewidth=0.5)

        ax1.set_box_aspect(1)
        ax2.set_box_aspect(1)

    # @State.static_dataset_decorator
    @context_manager_decorator
    def plot_hysteresis_waveform(
        self, fig, ax, inset_pos, x_start, x_end, y_inset_min=-2, y_inset_max=20
    ):
        # Plot the hysteresis waveform and add a zoomed-in inset
        ax.plot(self.waveform_constructor())
        ax_new = ax.inset_axes(inset_pos)
        ax_new.plot(self.waveform_constructor())
        ax_new.set_xlim(x_start, x_end)
        ax_new.set_ylim(-2, 20)

        # Draw the inset connector lines
        inset_connector(
            fig,
            ax,
            ax_new,
            [(x_start, y_inset_min), (x_end, y_inset_min)],
            [(x_start, y_inset_min), (x_end, y_inset_min)],
            color="k",
            linestyle="--",
            linewidth=0.5,
        )

        # Add a box around the inset area on the main plot
        add_box(
            ax,
            (x_start, y_inset_min, x_end, y_inset_max),
            edgecolor="k",
            linestyle="--",
            facecolor="none",
            linewidth=0.5,
            zorder=10,
        )

        ax.set_xlabel("Voltage Steps")
        ax.set_ylabel("Voltage (V)")

    # @State.static_dataset_decorator
    @context_manager_decorator
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
        pixel = np.random.randint(0, self.num_pix)
        voltagestep = np.random.randint(0, self.voltage_steps)

        print("pixel: ", pixel)
        print("voltagestep: ", voltagestep)

        # Initialize the figure and axes for plotting
        fig, ax = layout_fig(5, 5, figsize=figsize)

        # Calculate the number of voltage steps in one BE waveform cycle
        be_voltagesteps = len(self.be_waveform) / self.be_repeats

        # Plot the BE waveform
        ax[0].plot(self.be_waveform[: int(be_voltagesteps)])
        ax[0].set(xlabel="Time (sec)", ylabel="Voltage (V)")

        # Perform Fourier Transform on the BE waveform to get the resonance graph
        resonance_graph = np.fft.fft(self.be_waveform[: int(be_voltagesteps)])
        fftfreq = fftpack.fftfreq(int(be_voltagesteps)) * self.sampling_rate

        # Plot the resonance graph
        ax[1].plot(
            fftfreq[: int(be_voltagesteps) // 2],
            np.abs(resonance_graph[: int(be_voltagesteps) // 2]),
        )
        ax[1].axvline(
            x=self.be_center_frequency,
            ymax=np.max(resonance_graph[: int(be_voltagesteps) // 2]),
            linestyle="--",
            color="r",
        )
        ax[1].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")

        # Set the x-axis limits based on the BE center frequency and bandwidth
        ax[1].set_xlim(
            self.be_center_frequency - self.be_bandwidth - self.be_bandwidth * 0.25,
            self.be_center_frequency + self.be_bandwidth + self.be_bandwidth * 0.25,
        )

        self.plot_hysteresis_waveform(fig, ax[2], inset_pos, x_start, x_end)

        # Set the dataset state to retrieve the magnitude spectrum
        self.scaled = False
        self.raw_format = "magnitude spectrum"
        self.measurement_state = "all"
        self.resampled = False

        # Get the magnitude spectrum for the selected pixel and voltage step
        # JGoddy added scaled on May 14 without testing it
        # if 'scaled' should always be False here, then remove it from the function call
        data_ = self.raw_spectra(pixel, voltagestep, scaled = self.scaled)

        # Plot the magnitude spectrum
        ax[3].plot(
            self.frequency_bin,
            data_[0].flatten(),
        )
        ax[3].set(
            xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)", facecolor="none"
        )

        # Plot the phase spectrum on the same plot with a secondary y-axis
        ax2 = ax[3].twinx()
        ax2.plot(
            self.frequency_bin,
            data_[1].flatten(),
            "r",
        )
        ax2.set(xlabel="Frequency (Hz)", ylabel="Phase (rad)")
        ax[3].set_zorder(ax2.get_zorder() + 1)

        # Switch the dataset back to complex format
        self.raw_format = "complex"
        # JGoddy added scaled on May 14 without testing it
        # if 'scaled' should always be False here, then remove it from the function call
        data_ = self.raw_spectra(pixel, voltagestep, scaled = self.scaled)

        # Plot the real and imaginary components of the spectra
        ax[4].plot(self.frequency_bin, data_[0].flatten(), label="Real")
        ax[4].set(xlabel="Frequency (Hz)", ylabel="Real (Arb. U.)")
        ax3 = ax[4].twinx()
        ax3.plot(self.frequency_bin, data_[1].flatten(), "r", label="Imaginary")
        ax3.set(xlabel="Frequency (Hz)", ylabel="Imag (Arb. U.)", facecolor="none")

        set_sci_notation_label(ax[1], axis="x", corner="bottom right")
        set_sci_notation_label(ax[2], axis="x", corner="bottom right")
        set_sci_notation_label(ax[3], axis="x", corner="bottom right")
        set_sci_notation_label(ax[4], axis="x", corner="bottom right")

        # Save the figure if a printer object is available
        if self.printer is not None:
            self.printer.savefig(fig, filename, label_figs=ax, style="b")

        plt.close(fig)
        return fig

    # @State.static_scale_decorator
    @context_manager_decorator
    def SHO_hist(self, SHO_data, filename=None, scaled=False, loc='tr', inset_fraction=(0.15,0.15), style="b",**kwargs):
        """Plots the SHO hysteresis parameters

        Args:
            SHO_data (numpy): SHO fit results
            filename (str, optional): filename where to save the results. Defaults to "".
        """

        # from matplotlib.ticker import ScalarFormatter

        # xfmt = ScalarFormatter()
        # xfmt.set_powerlimits()  # Or whatever your limits are . . .

        # if the scale is False will not use the scale in the viz
        if self.scaled or scaled:
            print("dataset is scaled")
            self.SHO_ranges = None

        # if the SHO data is not a list it will make it a list
        if type(SHO_data) is not list:
            SHO_data = [SHO_data]

        # check distributions of each parameter before and after scaling
        fig, axs = layout_fig(
            4 * len(SHO_data),
            4,
            figsize=(15, 1.25 * len(SHO_data)),  # figsize=(5.25, 1.25 * len(SHO_data))
        )

        for k, SHO_data_ in enumerate(SHO_data):
            axs_ = axs[k * 4 : (k + 1) * 4]

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
                # ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0),useMathText=True)
                # ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0),useMathText=True)

                set_sci_notation_label(ax, axis="x", corner="bottom right")
                set_sci_notation_label(ax, axis="y", corner="top left")

                ax.xaxis.labelpad = 0  # 10

                ax.set_box_aspect(1)

            if self.verbose:
                self.extraction_state

        # prints the figure
        if self.printer is not None and filename is not None: 
            self.printer.savefig(fig, filename, label_figs=axs, loc=loc, inset_fraction=inset_fraction, style=style, **kwargs)

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
        the specified filename if a printer object is available.
        """

        if data is None:
            # If no data is provided, select a random pixel from the dataset
            pixel = np.random.randint(0, self.num_pix)
            data = self.SHO_fit_results()[[pixel], :, :]

        # Initialize the figure and axes with a 4x4 grid layout
        fig, axs = layout_fig(4, 4, figsize=(5.5, 1.1))

        # Loop over each axis and corresponding SHO label to plot the fit results
        for i, (ax, label) in enumerate(zip(axs, self.SHO_labels)):
            ax.plot(self.dc_voltage, data[0, :, i])
            ax.set_ylabel(label["y_label"])

            if abs(int("{:.1e}".format(np.min(data[0, :, i])).split("e")[1])) > 2:
                set_sci_notation_label(ax, axis="y", corner="top left")

            if i == 3:
                ax.set_yticks([3, 0, -3])

            ax.set_xticks([-15, 0, 15])

        # If verbose mode is enabled, log the current extraction state (for debugging or tracking)
        if self.verbose:
            self.extraction_state

        # If a printer object is defined, save the figure with the specified filename and style
        if self.printer is not None:
            self.printer.savefig(fig, filename, label_figs=axs, loc = 'tr',inset_fraction = (0.15, 0.15), style="b")

    ###### MOVIES #####

    # @State.static_dataset_decorator
    @context_manager_decorator
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
        self.set_attributes(**output_state)

        # Constructs the basepath for saving images if provided
        if basepath is not None:
            # If a model path is provided, name the directory based on the model with the lowest loss
            if model_path is not None:
                model_filename = (
                    model_path
                    + "/"
                    + self.get_lowest_loss_for_noise_level(model_path, noise)
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
        names = ["A", "\u03c9", "Q", "\u03c6"]

        # Retrieves the DC voltage data (only for the "on" state)
        voltage = self.dc_voltage

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
            ax[0].plot(self.dc_voltage, "k")
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
                        bar_ax.append(fig.add_axes(fig_scalar.to_relative(pos_inch)))

                        # Add the colorbar to the axis
                        fmt = ScalarFormatter(useMathText=True)
                        fmt.set_powerlimits((0, 0))
                        cbar = plt.colorbar(
                            ax[i + 1].images[0],
                            cax=bar_ax[i],
                            format=fmt,
                            ticks=np.linspace(
                                self.SHO_ranges[i][0], self.SHO_ranges[i][1], 5
                            ),
                        )

                        cbar.set_label(names[i])  # Label the colorbar

            # Add a scalebar to the last axis if specified
            if self.image_scalebar is not None:
                scalebar(ax[-1], *self.image_scalebar)

            # Save the figure if a printer object and filename are provided
            if self.printer is not None and filename is not None:
                self.printer.savefig(
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
        pos_inch[1] -= (
            embedding_image_size + 0.33 * inter_gap_count
        )  # Adjust bottom position

        # Set the size for embedding images
        pos_inch[2] = embedding_image_size  # Width of embedding image
        pos_inch[3] = embedding_image_size  # Height of embedding image

        # Loop through the rows to add the subplots for the embedding images
        for j in range(rows):
            # Add 4 graphs per row
            for i in range(4):
                ax.append(
                    fig.add_axes(fig_scalar.to_relative(pos_inch))
                )  # Add subplot to figure

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
                ax_.extend(ax[1 + 2 * i + 8 * j : 3 + 2 * i + 8 * j])
                ax_.extend(ax[5 + 2 * i + 8 * j : 7 + 2 * i + 8 * j])

        return fig, ax_, fig_scalar

    # @static_dataset_decorator
    @context_manager_decorator
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
            pixel = np.random.randint(0, self.num_pix)

        # Get the appropriate voltage step, considering the current state
        voltage_step = self.get_voltage_step(voltage_step)

        # Set object attributes based on the predict dictionary
        self.set_attributes(**predict)

        # Compute the fit parameters for the selected pixel and voltage step
        params = self.SHO_LSQF(pixel=pixel, voltage_step=voltage_step)

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

    # @static_dataset_decorator
    @context_manager_decorator
    def nn_checker(
        self, state, filename=None, pixel=None, voltage_step=None, legend=True, **kwargs
    ):
        # if a pixel is not provided it will select a random pixel
        if pixel is None:
            # Select a random point and time step to plot
            pixel = np.random.randint(0, self.num_pix)

        # gets the voltagestep with consideration of the current state
        voltage_step = self.get_voltage_step(voltage_step)

        self.set_attributes(**state)

        # JGoddy added scaled on May 14 without testing it
        # if 'scaled' should always be False here, then remove it from the function call
        data = self.raw_spectra(pixel=pixel, voltage_step=voltage_step, scaled = self.scaled)

        # plot real and imaginary components of resampled data
        fig = plt.figure(figsize=(3, 1.25), layout="compressed")
        axs = plt.subplot(111)

        self.raw_format = "complex"

        # JGoddy added scaled on May 14 without testing it
        # if 'scaled' should always be False here, then remove it from the function call
        data, x = self.raw_spectra(pixel, voltage_step, frequency=True, scaled = self.scaled, **kwargs)

        axs.plot(x, data[0].flatten(), "k", label=self.label + " Real")
        axs.set_xlabel("Frequency (Hz)")
        axs.set_ylabel("Real (Arb. U.)")
        ax2 = axs.twinx()
        ax2.set_ylabel("Imag (Arb. U.)")
        ax2.plot(x, data[1].flatten(), "g", label=self.label + " Imag")
        self._scientific_notation_dual(axs, ax2)

        axes = [axs, ax2]

        for ax in axes:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.set_box_aspect(1)

        if self.verbose:
            self.extraction_state

        if legend:
            fig.legend(bbox_to_anchor=(1.0, 1), loc="upper right", borderaxespad=0.1)

        # prints the figure
        if self.printer is not None and filename is not None:
            self.printer.savefig(fig, filename, style="b")

    ##### Analytics #####

    # @static_dataset_decorator
    @context_manager_decorator
    def bmw_nn(
        self,
        true_state,
        prediction=None,
        model=None,
        out_state=None,
        n=1,
        gaps=(0.8, 0.4),
        size=(1.25, 1.25),
        filename=None,
        compare_state=None,
        fit_type="SHO",
        **kwargs,
    ):
        # TODO: I had to remove this for fitting --
        # true_state = torch.atleast_3d(torch.tensor(true_state.reshape(-1,96)))

        d1, d2, x1, x2, label, index1, mse1 = None, None, None, None, None, None, None

        if fit_type == "SHO":
            d1, d2, x1, x2, label, full_indices, index1, mse1 = (
                self.get_best_median_worst(
                    true_state,
                    prediction=prediction,
                    model=model,
                    out_state=out_state,
                    n=n,
                    compare_state=compare_state,
                    **kwargs,
                )
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

                ax_.set_xlabel("Frequency (Hz)", labelpad=0)

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
                            ax_.set_ylabel("Amplitude (Arb. U.)", labelpad=1)
                            ax1.set_ylabel("Phase (rad)", labelpad=1)
                    else:
                        ax_.set_ylabel("Real (Arb. U.)", labelpad=1)
                        ax1.set_ylabel("Imag (Arb. U.)", labelpad=1)

                self._scientific_notation_dual(ax_, ax1)

            # add a legend just for the last one
            lines, labels = ax_.get_legend_handles_labels()
            lines2, labels2 = ax1.get_legend_handles_labels()
            ax_.legend(lines + lines2, labels + labels2, loc="upper right")

        elif fit_type == "hysteresis":
            d1, d2, x1, x2, label, full_indices, index1, mse1 = (
                self.get_best_median_worst(
                    true_state,
                    prediction=prediction,
                    n=n,
                    **kwargs,
                    fit_type=fit_type,
                )
            )

            fig, ax = subfigures(1, 3, gaps=gaps, size=size)

            for i, (true, prediction, error) in enumerate(zip(d1, d2, mse1)):
                ax_ = ax[i]

                # unscale the hysteresis loops for plotting
                prediction = self.hysteresis_scaler.inverse_transform(prediction)
                true = self.hysteresis_scaler.inverse_transform(true)

                ax_.plot(
                    x2,
                    prediction,
                    color=color_palette["NN_A"],
                    # label=f"NN {label[0]}",
                )

                ax_.plot(
                    x1,
                    true,
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

                set_sci_notation_label(ax_, axis="y", corner="top left")

        else:
            raise ValueError("fit_type must be SHO or hysteresis")

        # prints the figure
        if self.printer is not None and filename is not None:
            self.printer.savefig(fig, filename, label_figs=ax, style="b")

        if "returns" in kwargs.keys():
            if kwargs["returns"] == True:
                return d1, d2, index1, mse1

   


    # @static_dataset_decorator
    @context_manager_decorator
    def SHO_switching_maps(
        self,
        SHO_,
        colorbars=True,
        clims=[
            (0, 1.4e-4),  # amplitude
            (1.31e6, 1.33e6),  # resonance frequency
            (-230, -160),  # quality factor
            (-np.pi, np.pi),
        ],  # phase
        measurement_state="off",  # sets the measurement state to get the data
        cycle=2,  # sets the cycle to get the data
        cols=3,
        fig_width=6.5,  # figure width in inches
        number_of_steps=9,  # number of steps on the graph
        voltage_plot_height=1.25,  # height of the voltage plot
        intra_gap=0.02,  # gap between the graphs,
        inter_gap=0.05,  # gap between the graphs,
        cbar_gap=0.4,  # gap between the graphs of colorbars
        cbar_space=1.3,  # space on the right where the cbar is not
        filename=None,
        labels=None,
        label_marker_symbols_for_plt = ["o", "v", "^", ">", "<", "s","P", "D","*"],
        label_marker_size = 8,
        label_marker_starting_index = 0,
        label_letter_text_size=12,
    ):
        if type(SHO_) is not list:
            SHO_ = [SHO_]

        comp_number = len(SHO_)

        # sets the voltage state to off, and the cycle to get
        self.measurement_state = measurement_state
        self.cycle = cycle

        # instantiates the list of axes
        ax = []

        # number of rows
        rows = np.ceil(number_of_steps * comp_number / 3)

        # calculates the size of the embedding image
        embedding_image_size = (
            fig_width
            - (inter_gap * (cols - 1))
            - intra_gap * 3 * cols
            - cbar_space * colorbars
        ) / (cols * 4)

        # calculates the figure height based on the image details
        fig_height = (
            rows * (embedding_image_size + inter_gap)
            + voltage_plot_height
            + 0.33
            + inter_gap * (comp_number - 1)
        )

        # defines a scalar to convert inches to relative coordinates
        fig_scalar = FigDimConverter((fig_width, fig_height))

        # creates the figure
        fig = plt.figure(figsize=(fig_width, fig_height))

        # left bottom width height
        pos_inch = [
            0.33,
            fig_height - voltage_plot_height,
            fig_width - 0.33,
            voltage_plot_height,
        ]

        # adds the plot for the voltage
        ax.append(fig.add_axes(fig_scalar.to_relative(pos_inch)))

        # resets the x0 position for the embedding plots
        pos_inch[0] = 0
        pos_inch[1] -= embedding_image_size + 0.33+0.1

        # sets the embedding size of the image
        pos_inch[2] = embedding_image_size
        pos_inch[3] = embedding_image_size

        # This makes the figures
        for k, _SHO in enumerate(SHO_):
            # adds the embedding plots
            for i in range(number_of_steps):
                # loops around the amp, phase, and freq
                for j in range(4):
                    # adds the plot to the figure
                    ax.append(fig.add_axes(fig_scalar.to_relative(pos_inch)))

                    # adds the inter plot gap
                    pos_inch[0] += embedding_image_size + intra_gap

                # if the last column in row, moves the position to the next row
                if (i + 1) % cols == 0 and i != 0:
                    # resets the x0 position for the embedding plots
                    pos_inch[0] = 0

                    # moves the y0 position to the next row
                    pos_inch[1] -= embedding_image_size + inter_gap

                    if (i + 1) % (cols * comp_number) == 0 and comp_number > 1:
                        pos_inch[1] -= inter_gap

                else:
                    # adds the small gap between the plots
                    pos_inch[0] += inter_gap

        # gets the DC voltage data - this is for only the on state or else it would all be 0
        # voltage = self.dataset.dc_voltage

        # # gets just part of the loop
        # if hasattr(self.dataset, "cycle") and self.dataset.cycle is not None:
        #     # gets the cycle of interest
        #     voltage = self.dataset.get_cycle(voltage)

        voltage = np.swapaxes(np.atleast_2d(self.get_voltage), 0, 1).astype(np.float64)
        voltage = self.roll_hysteresis(voltage)

        # gets the index of the voltage steps to plot
        inds = np.linspace(0, len(voltage) - 1, number_of_steps, dtype=int)

        # plots the voltage
        ax[0].plot(voltage, "k", linewidth=0.5)
        ax[0].set_ylabel("Voltage (V)",fontsize=12)
        ax[0].set_xlabel("Step",fontsize=12)
        ax[0].tick_params(axis='x',labelsize=8)
        ax[0].tick_params(axis='y',labelsize=8)
        ax[0].set_xticks(np.linspace(0,100,11))
        ax[0].set_yticks([-15,0,15])

        #label_marker_symbols_for_plt = ["o", "v", "^", ">", "<", "s","P", "D","*"]

        # Plot the data with different markers
        for i, ind in enumerate(inds):
            # this adds the labels to the graphs
            ax[0].plot(ind, voltage[ind], label_marker_symbols_for_plt[i], color="k", markersize=label_marker_size)
            vshift = (ax[0].get_ylim()[1] - ax[0].get_ylim()[0]) * 0.25

            # positions the location of the labels
            if voltage[ind] - vshift - 0.15 < ax[0].get_ylim()[0]:
                vshift = -vshift / 2

            # adds the text to the graphs
            ax[0].text(ind, voltage[ind] - 0.75*vshift, number_to_letters(i + label_marker_starting_index), color="k", fontsize=label_letter_text_size)

        for k, _SHO in enumerate(SHO_):
            # converts the data to a numpy array
            if isinstance(_SHO, torch.Tensor):
                _SHO = _SHO.detach().numpy()

            #print(_SHO.shape)
            _SHO = _SHO.reshape(self.num_pix, self.voltage_steps, 4)

            # get the selected measurement cycle
            _SHO = self.get_measurement_cycle(_SHO, axis=1)

            names = ["A", "\u03c9", "Q", "\u03c6"]

            for i, ind in enumerate(inds):
                axis_start = int(
                    (i % cols) * 4
                    + ((i) // cols) * (comp_number * cols * 4)
                    + k * (cols * 4)
                    + 1
                )

                # loops around the amp, resonant frequency, and Q, Phase
                for j in range(4):
                    imagemap(
                        ax[axis_start + j],
                        _SHO[:, ind, j],
                        colorbars=False,
                        cmap="viridis",
                    )

                    if i // rows == 0 and k == 0:
                        labelfigs(
                            ax[axis_start + j],
                            string_add=names[j],
                            loc="cb",
                            label_size=5,
                            inset_fraction=(0.2, 0.2),
                        )

                    ax[axis_start + j].images[0].set_clim(clims[j])

                    if k == 0:
                        labelfigs(
                            ax[axis_start + j],
                            string_add=str(i + 1),
                            label_size=5,
                            loc="bl",
                            inset_fraction=(0.2, 0.2),
                        )

                    if labels is not None and (axis_start + j) % (4 * cols) == 1:
                        ax[axis_start + j].set_ylabel(labels[k])

        # if add colorbars
        if colorbars:
            # builds a list to store the colorbar axis objects
            bar_ax = []

            # gets the voltage axis position in ([xmin, ymin, xmax, ymax]])
            voltage_ax_pos = fig_scalar.to_inches(
                np.array(ax[0].get_position()).flatten()
            )

            fmt = ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((0, 0))
            # loops around the 4 axis
            for i in range(4):
                # calculates the height and width of the colorbars
                cbar_h = (voltage_ax_pos[1] - inter_gap - 2 * intra_gap - 0.33) / 2
                cbar_w = (cbar_space - inter_gap - 2 * cbar_gap) / 2

                # sets the position of the axis in inches
                pos_inch = [
                    voltage_ax_pos[2] - (2 - i % 2) * (cbar_gap + cbar_w) + inter_gap,
                    voltage_ax_pos[1] - (i // 2) * (inter_gap + cbar_h) - 0.33 - cbar_h,
                    cbar_w - 0.02,
                    cbar_h - 0.1,
                ]

                # adds the plot to the figure
                bar_ax.append(fig.add_axes(fig_scalar.to_relative(pos_inch)))

                # adds the colorbars to the plots
                fmt = ScalarFormatter(useMathText=True)
                fmt.set_powerlimits((0, 0))
                cbar = plt.colorbar(ax[i + 1].images[0], cax=bar_ax[i], format=fmt)
                cbar.set_label(names[i])  # Add a label to the colorbar
                

        # prints the figure
        if self.printer is not None and filename is not None:
            self.printer.savefig(
                fig, filename, size=6, loc="tl", inset_fraction=(0.2, 0.2)
            )
        plt.close(fig)
        return fig

    # @static_dataset_decorator
    @context_manager_decorator
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
        pixel, voltage = np.unravel_index(index, (self.num_pix, self.voltage_steps))

        # Case 1: The model is a neural network (nn.Module)
        if isinstance(model, nn.Module):
            # Retrieve the input data for the neural network
            X_data, Y_data = self.get_nn_data()

            # Select the data based on the provided indices
            X_data = X_data[[index]]

            # Use the model to predict the data and SHO parameters
            pred_data, scaled_param, params = model.predict(X_data)

            # Convert the predicted data to a NumPy array
            pred_data = np.array(pred_data)

        # Case 2: The model is a dictionary (assumed to be an LSQF model)
        if isinstance(model, dict):
            # Ensure that the dataset is not scaled when retrieving raw parameters
            self.scaled = False

            # Retrieve the SHO fit results without any phase shift
            params_shifted = self.SHO_fit_results()

            # Ensure the phase shift for the current fitter is set to zero
            exec(f"self.{model['fitter']}_phase_shift = 0")

            # Retrieve the SHO fit parameters
            params = self.SHO_fit_results()

            # Switch back to scaled parameters for further processing
            self.scaled = True

            # Generate raw spectra from the fit results
            pred_data = self.raw_spectra(fit_results=params,
                                         voltage_step=self.get_voltage_step(),
                                        frequency=False,
                                        scaled=self.scaled)

            # Reshape the predicted data for correct dimensionality (samples, channels, voltage steps)
            pred_data = np.array(
                [pred_data[0], pred_data[1]]
            )  # (channels, samples, voltage steps)
            pred_data = np.swapaxes(
                pred_data, 0, 1
            )  # (samples, channels, voltage steps)
            pred_data = np.swapaxes(
                pred_data, 1, 2
            )  # (samples, voltage steps, channels)

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

    # @static_dataset_decorator
    @context_manager_decorator
    def get_mse_index(self, index, model):
        """
        Computes the Mean Squared Error (MSE) between the raw spectra data and the predicted data
        for a given set of indices and a specified model.

        This function retrieves the raw data from the dataset and compares it with the predicted
        data from the provided model. Depending on whether the model is a neural network (`nn.Module`)
        or an LSQF model (represented as a dictionary), it handles predictions accordingly and
        calculates the MSE.

        Args:
            index (list): List of indices specifying which samples to compute the MSE for.
            model (any): Model used to generate predictions. Can either be:
                        - A neural network (`nn.Module`), in which case predictions are obtained from the model.
                        - A dictionary representing an LSQF model, where predictions are computed using SHO fitting.

        Returns:
            float: The computed Mean Squared Error (MSE) between the raw data and the predicted data.

        Notes:
            - For neural network models (`nn.Module`), predictions are obtained directly from the model.
            - For LSQF models, the raw spectra are generated using the unscaled SHO parameters.
        """

        # Retrieve the raw dataset (samples, voltage steps, real/imaginary)
        data, _ = self.get_nn_data()

        # Select the data for the given indices
        data = data[[index]]

        # Case 1: Model is a neural network (nn.Module)
        if isinstance(model, nn.Module):
            # Get the predictions from the neural network model
            predictions, params_scaled, params = model.predict(data)

            # Detach the predictions tensor from the computational graph and convert to NumPy array
            predictions = predictions.detach().numpy()

        # Case 2: Model is an LSQF model (represented as a dictionary)
        if isinstance(model, dict):
            # Set the phase shift for the specific fitter to zero (required for proper fitting)
            exec(f"self.{model['fitter']}_phase_shift = 0")

            # Disable scaling to get unscaled SHO parameters (needed for generating raw data)
            self.scaled = False

            # Retrieve the SHO fit results (parameters)
            params = self.SHO_fit_results()

            # Re-enable scaling (since the MSE is calculated using scaled parameters)
            self.scaled = True

            # Ensure the measurement state is set to 'complex' format (for real/imaginary data)
            self.raw_format = "complex"

            # Generate raw spectra using the retrieved SHO parameters
            # JGoddy added scaled on May 14 without testing it
            # if 'scaled' should always be False here, then remove it from the function call
            pred_data = self.raw_spectra(fit_results=params, scaled = self.scaled)

            # Convert the predicted data to a NumPy array
            pred_data = np.array(
                pred_data
            )  # Shape: (real/imaginary, samples, voltage steps)

            # Roll the axes to match the required shape: (samples, voltage steps, real/imaginary)
            pred_data = np.rollaxis(pred_data, 0, pred_data.ndim)

            # Select the predicted data for the given indices
            predictions = pred_data[[index]]

        # Compute and return the MSE between the raw data and the predicted data
        return MSE(data.detach().numpy(), predictions)

    # @static_dataset_decorator
    @context_manager_decorator
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
            - The function supports saving the generated figure to a file using the `printer` object.
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
        list_ax_ = []
        list_ax1_ = [] 

        # Loop through each fit and the associated data
        for step, (data, name) in enumerate(zip(data, names)):
            # Unpack the data (true, predicted values, indices, etc.)
            d1, d2, x1, x2, label, full_labels, index1, mse1, params = data

            # Loop through datasets for comparison (true vs. predicted data)
            for bmw, (true, prediction, error, SHO, index1) in enumerate(
                zip(d1, d2, mse1, params, index1)
            ):
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
                        errors[color] = self.get_mse_index(
                            index1, model_comparison[step]
                        )
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
                            # error_string = f"MSE - LSQF: {errors['LSQF']:0.4f} NN: {errors['NN']:0.4f}\n AMP - LSQF: {SHOs['LSQF'][0]:0.2e} NN: {SHOs['NN'][0]:0.2e}\n\u03c9 - LSQF: {SHOs['LSQF'][1]/1000:0.1f} NN: {SHOs['NN'][1]/1000:0.1f} Hz\nQ - LSQF: {SHOs['LSQF'][2]:0.1f} NN: {SHOs['NN'][2]:0.1f}\n\u03c6 - LSQF: {SHOs['LSQF'][3]:0.2f} NN: {SHOs['NN'][3]:0.1f} rad"
                            error_string = f"MSE - LSQF: {errors['LSQF']:0.4f} NN: {errors['NN']:0.4f}\n AMP - LSQF: {format(SHOs['LSQF'][0], '0.2e').split('e')[0]}$\\times10^{'{'}{format(SHOs['LSQF'][0], '0.2e').split('e')[-1]}{'}'}$ NN: {format(SHOs['LSQF'][0], '0.2e').split('e')[0]}$\\times10^{'{'}{format(SHOs['NN'][0], '0.2e').split('e')[-1]}{'}'}$ \n\u03c9 - LSQF: {SHOs['LSQF'][1] / 1000:0.1f} NN: {SHOs['NN'][1] / 1000:0.1f} Hz\nQ - LSQF: {SHOs['LSQF'][2]:0.1f} NN: {SHOs['NN'][2]:0.1f}\n\u03c6 - LSQF: {SHOs['LSQF'][3]:0.2f} NN: {SHOs['NN'][3]:0.1f} rad"

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
                    if (
                        "raw_format" in out_state.keys()
                        and out_state["raw_format"] == "magnitude spectrum"
                    ):
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

                set_sci_notation_label(ax_, axis="x", corner="bottom right")
                set_sci_notation_label(ax_, axis="y", corner="top left")
                
                list_ax_.append(ax_)
                list_ax1_.append(ax1)


        # Save the figure if filename is provided
        if self.printer is not None and filename is not None:
            self.printer.savefig(fig, filename, label_figs=ax, style="b")

        return fig,list_ax_,list_ax1_

    def hysteresis_comparison(self,
                             data,
                             row=None,
                             col=None,
                             cycle=None,
                             size=(1.25, 1.25),
                             gaps=(1, 0.66),
                             nn_model=None,
                             measurement_state=None,
                             filename="hysteresis_comparison"):
        """
        Plot a comparison of the hysteresis loop.

        Args:
            data (list): List of data types to plot.
            row (int, optional): Row to plot. Defaults to None.
            col (int, optional): Column to plot. Defaults to None.
            cycle (int, optional): Cycle to plot. Defaults to None.
            size (tuple, optional): Size of the image to plot. Defaults to (1.25, 1.25).
            gaps (tuple, optional): Gaps between subplots. Defaults to (1, 0.66).
            nn_model (object, optional): Neural network model for comparison. Defaults to None.
            measurement_state (str, optional): Measurement state to plot. Defaults to None.
            filename (str, optional): Filename to save the plot. Defaults to "hysteresis_comparison".
        """

        # sets the measurement state
        if self.measurement_state is not None:
            self.measurement_state = measurement_state

        # if only the LSQF is to be plotted
        if 'LSQF' in data and 'NN' not in data:
            # gets the LSQF Hysteresis Loops from the Dataset
            loops, raw_hysteresis_loop_scaled, voltage = self.get_LSQF_hysteresis_fits(compare=True, index=False)

            raw_hysteresis_loop = self.hysteresis_scaler.inverse_transform(raw_hysteresis_loop_scaled)

            # selects a point to plot
            row, col, cycle = self.get_selected_hysteresis(
                raw_hysteresis_loop, row, col, cycle)

            self.random_hysteresis(raw_hysteresis_loop,
                                   loops,
                                   voltage,
                                   filename,
                                   size,
                                   row, col, cycle)
            return

        # gets the LSQF Hysteresis Loops from the Dataset
        loops, raw_hysteresis_loop_scaled, voltage = self.get_LSQF_hysteresis_fits(compare=True)

        # scales the loops for comparison
        loops_scaled = self.hysteresis_scaler.transform(loops)
        raw_hysteresis_loop = self.hysteresis_scaler.inverse_transform(raw_hysteresis_loop_scaled)

        # gets the NN data for comparison
        if nn_model is not None:
            # gets the data for model prediction with the NN
            _data, voltage = self.get_hysteresis(scaled=True, loop_interpolated=True)
            _data = torch.atleast_3d(torch.tensor(_data.reshape(-1, self.voltage_steps_per_cycle))).float()

            NN_pred_data, NN_scaled_params, NN_params = nn_model.predict(
                _data, translate_params=False, is_SHO=False)
            NN_loops = self.hysteresis_function(y=NN_params, V=voltage[:, 0].squeeze()).to(
                'cpu').detach().numpy().squeeze()
            NN_loops_scaled = self.hysteresis_scaler.transform(NN_loops)

        # if we are plotting the NN and LSQF results
        fig, ax = subfigures(3, len(data), gaps=gaps, size=size)

        # loops around the models provided
        for j, model in enumerate(data):

            if model == 'LSQF':
                out = self.ranked_mse(raw_hysteresis_loop_scaled,
                                      {'LSQF': loops_scaled},
                                      {'NN': NN_loops_scaled})

            elif model == 'NN':
                out = self.ranked_mse(raw_hysteresis_loop_scaled,
                                      {'NN': NN_loops_scaled},
                                      {'LSQF': loops_scaled})

            for i, results in enumerate(out):

                # sets the index for the plots
                plot_idx = i * 2 + j

                index = int(results['Original Index'])

                ax[plot_idx].plot(voltage,
                                  raw_hysteresis_loop[index], 'o', label="Raw Data")

                ax[plot_idx].plot(voltage,
                                  loops[index], 'r', label='LSQF')

                ax[plot_idx].plot(voltage,
                                  NN_loops[index], 'g', label='NN')

                #ax[plot_idx].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                
                set_sci_notation_label(ax[plot_idx], axis = "y", corner = 'top left',textsize = 6)


                # Position text at (1 inch, 2 inches) from the bottom left corner of the figure
                text_position_in_inches = (
                    -1 * (gaps[0] + size[0]) * ((2 - i) % 3) + size[0] / 2,
                    (gaps[1] + size[1]) * (1.25 - i // 3 - 1.25) - gaps[1],
                )

                # gets the axis position in inches - gets the bottom center
                center = get_axis_pos_inches(fig, ax[plot_idx])

                # selects the text position as an offset from the bottom center
                text_position_in_inches = (center[0], center[1] - 0.32 + .125)

                error = results['MSE_LSQF']

                error_string = f"LSQF MSE: {error:0.4f}"

                add_text_to_figure(
                    fig,
                    error_string,
                    text_position_in_inches,
                    fontsize=6,
                    ha="center",
                )

                # selects the text position as an offset from the bottom center
                text_position_in_inches = (center[0], center[1] - 0.3)

                error = results['MSE_NN']

                error_string = f"NN MSE: {error:0.4f}"

                add_text_to_figure(
                    fig,
                    error_string,
                    text_position_in_inches,
                    fontsize=6,
                    ha="center",
                )

                ax[plot_idx - 1].set_ylabel("(Arb. U.)")
                ax[plot_idx].set_ylabel("(Arb. U.)")

        # add a legend just for the last one
        lines, labels = ax[plot_idx - 1].get_legend_handles_labels()
        ax[plot_idx - 1].legend(lines, labels, loc="upper right")
        lines, labels = ax[plot_idx].get_legend_handles_labels()
        ax[plot_idx].legend(lines, labels, loc="upper right")

        ax[plot_idx - 1].set_xlabel("Voltage (V)")
        ax[plot_idx].set_xlabel("Voltage (V)")
        ax[plot_idx - 1].xaxis.set_label_coords(0.5, -0.28)
        ax[plot_idx].xaxis.set_label_coords(0.5, -0.28)
        

        # prints the figure
        if self.printer is not None and filename is not None:
            self.printer.savefig(fig, filename, label_figs=ax, style="b")
        
        plt.close(fig)
        return fig



    # @static_dataset_decorator
    @context_manager_decorator
    def violin_plot_comparison_SHO(self, state, model, X_data, params=None, filename=None, label="NN",figlabel = 'a', ax=None, loc='tr', inset_fraction=(0.05,0.05), style="b",**kwargs):
        """
        Generates a violin plot to compare true parameter values obtained from the SHO LSQF fit
        and predicted parameter values from a machine learning model.

        Parameters:
        -----------
        state : dict
            A dictionary containing the necessary state attributes to configure the object.
        model : object
            A machine learning model that has a `predict` method to generate parameter predictions
            from input data.
        X_data : array-like
            Input data for the model to generate predictions.
        params : array-like
            Parameters for the model to generate predictions.
        filename : str
            Filename to save the generated plot. If None, the plot is not saved.
        label : str
            Label for the predicted dataset. Defaults to "NN".

        Returns:
        --------
        fig : matplotlib.figure.Figure
            A matplotlib figure object representing the violin plot.
        """
        # Set the object attributes using the provided state dictionary
        self.set_attributes(**state)

        # Initialize an empty dataframe to store the data for plotting
        df = pd.DataFrame()

        if params is None:
            # Use the model to get predicted parameter values and other outputs
            pred_data, scaled_param, params = model.predict(X_data)

        # Scale the predicted parameters using the SHO scaler
        scaled_param = self.SHO_scaler.transform(params)

        # Obtain the true parameter values from the SHO LSQF fit
        true = self.SHO_fit_results().reshape(-1, 4)

        # Create dataframes for true and predicted parameter values with appropriate column names
        true_df = pd.DataFrame(
            true, columns=["Amplitude", "Resonance", "Q-Factor", "Phase"]
        )
        predicted_df = pd.DataFrame(
            scaled_param, columns=["Amplitude", "Resonance", "Q-Factor", "Phase"]
        )

        # Concatenate the true and predicted dataframes into a single dataframe for plotting
        df = pd.concat((true_df, predicted_df))

        # Define the datasets and labels for the violin plot
        names = [true, scaled_param]
        names_str = ["LSQF", label]  # Labels for true and predicted datasets
        labels = [
            "A",
            "\u03c9",
            "Q",
            "\u03c6",
        ]  # Labels for parameters: Amplitude (A), Resonance (ω), Q-Factor (Q), Phase (φ)

        # Append parameter, value, and dataset information into the dataframe
        for j, name in enumerate(names):
            for i, label in enumerate(labels):
                dict_ = {
                    "value": name[:, i],  # Parameter values (true or predicted)
                    "parameter": np.repeat(
                        label, name.shape[0]
                    ),  # Parameter type (A, ω, Q, φ)
                    "dataset": np.repeat(
                        names_str[j], name.shape[0]
                    ),  # Dataset label (LSQF or NN)
                }
                df = pd.concat((df, pd.DataFrame(dict_)))

        df = df.reset_index(drop=False)

        # Initialize a figure for plotting
        if ax is None:
            fig, ax = plt.subplots(figsize=(2, 2))
        else:
            fig = ax.figure

        # Generate the violin plot, comparing true and predicted parameter distributions
        sns.violinplot(
            data=df,
            x="parameter",
            y="value",
            hue="dataset",
            split=True,
            ax=ax,
            inner ='quartile',
            linewidth=0.1,
        )

        # Customize the appearance of the plot
        if figlabel is not None:
            labelfigs(ax, string_add = figlabel, style="b",**kwargs)  # Apply custom labeling style to the plot
        ax.set_ylabel("Scaled SHO Results")  # Set the y-axis label
        ax.set_xlabel("")  # No label for x-axis

        # Modify the legend associated with the plot
        legend = ax.get_legend()
        legend.set_title("")

        # Save the plot if a filename and printer are provided
        if self.printer is not None and filename is not None:
            self.printer.savefig(fig, filename)
        plt.close(fig)
        return fig


    def violin_plot_comparison_hysteresis(self, model, X_data, filename,ax=None):
        """
        Generates a violin plot comparing hysteresis parameters from a neural network model
        prediction and the least squares fitting (LSQF) results.

        Args:
            model: Object
                Trained model with a `predict` method to generate predictions for the input data.
            X_data: array-like
                Input data for which the model will generate predictions.
            filename: str
                The filename where the figure will be saved if a Printer object is defined.

        Returns:
            None
        """

        # Initialize an empty DataFrame to store the results
        df = pd.DataFrame()

        # Use the model to predict the hysteresis parameters for the provided input data
        pred_data, scaled_param, params = model.predict(X_data, is_SHO=False)

        # Get the true parameters from the least squares fit (LSQF) for hysteresis
        # The hysteresis parameters are reshaped to a format of (-1, 9)
        true = self.LSQF_hysteresis_params().reshape(-1, 9)

        # Scale the true parameters using the same scaler applied during the model training
        true_scaled = self.loop_param_scaler.transform(true)

        # Create DataFrames for true and predicted parameters with appropriate column labels
        true_df = pd.DataFrame(
            true, columns=["a0", "a1", "a2", "a3", "a4", "b0", "b1", "b2", "b3"]
        )
        predicted_df = pd.DataFrame(
            scaled_param, columns=["a0", "a1", "a2", "a3", "a4", "b0", "b1", "b2", "b3"]
        )

        # Concatenate true and predicted DataFrames
        df = pd.concat((predicted_df, true_df))

        # Prepare labels and dataset names for the plot
        names = [true_scaled, scaled_param]
        names_str = ["NN", "LSQF"]
        labels = ["a0", "a1", "a2", "a3", "a4", "b0", "b1", "b2", "b3"]

        # Add each parameter and corresponding label to the DataFrame
        for j, name in enumerate(names):
            for i, label in enumerate(labels):
                dict_ = {
                    "value": name[:, i],  # Scaled parameter values
                    "parameter": np.repeat(
                        label, name.shape[0]
                    ),  # Label for the parameter
                    "dataset": np.repeat(
                        names_str[j], name.shape[0]
                    ),  # Label for dataset type (NN or LSQF)
                }
                df = pd.concat((df, pd.DataFrame(dict_)))

        # Reset index to handle potential duplicated columns or indices
        df = df.reset_index(drop=False)
        
        # Create the figure for plotting
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        else:
            fig = ax.figure

        # Plot the violin plot with split view for comparing true and predicted values
        sns.violinplot(
            data=df,
            x="parameter",
            y="value",
            hue="dataset",
            split=True,
            ax=ax,
            linewidth=0.1,
        )

        # Style the plot with labels
        #TODO: why does the plot have the label 'a' (from labelfigs) when it is a stand alone plot?
        #labelfigs(ax, 0, style="b",inset_fraction = (0.05,0.95))
        ax.set_ylabel("Scaled SHO Results")
        ax.set_xlabel("")

        # Remove the legend title
        legend = ax.get_legend()
        legend.set_title("")

        # Save the figure if a printer object and filename are provided
        if self.printer is not None and filename is not None:
            self.printer.savefig(fig, filename)
            
        return ax
    
    def hysteresis_maps(
        self,
        parms_pred,
        colorbars=True,
        cycle=3,
        fig_width=10.5,  # figure width in inches
        filename=None,
    ):
        # # reshape data:
        # if data.shape != 3:

        # calculates the size of the embedding image
        embedding_image_size = 60

        fig, axs = plt.subplots(
            2,
            9,
            figsize=(fig_width, 4),
            gridspec_kw={"height_ratios": [1, 1]},
        )

        self.hysteresis_maps_colorbar_labels = [
            'a0', 'a1', 'a2', 'a3', 'a4', 'b0', 'b1', 'b2', 'b3'
        ]

        # Titles for each row
        row_titles = ['Predicted Parameters', 'LSQF Parameters']


        parms_lsqf = self.LSQF_hysteresis_params()[:, :, cycle, :].reshape(-1, 9)
        parms_pred = parms_pred.reshape(embedding_image_size, embedding_image_size, 4, 9)[:, :, cycle, :].reshape(-1, 9)

        self.hysteresis_maps_clims = []

       

        string_add = 'a'

        for i in range(9):
            self.hysteresis_maps_clims.append(
                (
                    np.min(
                        [
                            parms_pred[:, i].min(),
                            parms_lsqf[:, i].min(),
                        ]
                    ),
                    np.max(
                        [
                            parms_pred[:, i].max(),
                            parms_lsqf[:, i].max(),
                        ]
                    ),
                )
            )

            axs[0, i].imshow(
                parms_pred[:, i].reshape(
                    embedding_image_size, embedding_image_size),
                cmap="viridis",
                vmin=self.hysteresis_maps_clims[i][0],
                vmax=self.hysteresis_maps_clims[i][1],
            )
            axs[0,i].set_xticklabels('')
            axs[0,i].set_yticklabels('')
            axs[1, i].imshow(
                parms_lsqf[:, i].reshape(
                    embedding_image_size, embedding_image_size),
                cmap="viridis",
                vmin=self.hysteresis_maps_clims[i][0],
                vmax=self.hysteresis_maps_clims[i][1],
            )
            axs[1,i].set_xticklabels('')
            axs[1,i].set_yticklabels('')

            if colorbars:
                
                # Create an axis divider for each subplot
                divider = make_axes_locatable(axs[1, i])
                # Append axes to the bottom of the divider with appropriate padding
                cax = divider.append_axes("bottom", size="5%", pad=0.25) 
                
                
                self.hysteresis_maps_fmt = ScalarFormatter(useMathText=True)
                self.hysteresis_maps_fmt.set_powerlimits((0, 0))
                cbar = plt.colorbar(axs[1,i].images[0],
                                    cax=cax, format=self.hysteresis_maps_fmt,orientation = 'horizontal')
                cbar.set_label(self.hysteresis_maps_colorbar_labels[i])  # Add a label to the colorbar

                
             
        labelfigs(axs[0,0],
        string_add='a',
        loc ='tl',
        label_size=8,
        inset_fraction=(0.2, 0.2)
        )
        labelfigs(axs[1,0],
        string_add='b',
        loc ='tl',
        label_size=8,
        inset_fraction=(0.2, 0.2)
        )

        # Calculate the vertical position for the row titles
        title_y_positions = [0.85, 0.5]  # You may need to adjust these values

        # Set the titles for each row using fig.text
        for i, title in enumerate(row_titles):
            fig.text(0.5, title_y_positions[i], title, ha='center',
                     va='center', fontsize=20, transform=fig.transFigure)


        # prints the figure
        if self.printer is not None and filename is not None:
            print('use printing function')
            self.printer.savefig(
                fig, filename, size=6, loc="tl", inset_fraction=(0.2, 0.2)
            )
        plt.close(fig)
        return fig
    
    def random_hysteresis(self,
                          raw_hysteresis_loop,
                          lsqf_hysteresis_loop,
                          voltage,
                          filename,
                          size,
                          row, col, cycle):


        fig, ax = subfigures(1, 1, size=size)

        ax[0].plot(voltage.squeeze(),
                    raw_hysteresis_loop[row, col, cycle, :].squeeze(), 'o', label="Raw Data")


        ax[0].plot(voltage.squeeze(),
                    lsqf_hysteresis_loop[row, col, cycle, :].squeeze(), 'r', label='LSQF')

        ax[0].set_xlabel('Voltage (V)')
        ax[0].set_ylabel('Amplitude (Arb. U.)')
        set_sci_notation_label(ax[0], axis = "y", corner = 'top left')
        ax[0].legend()

        # prints the figure
        if self.printer is not None and filename is not None:
            self.printer.savefig(fig, filename, label_figs=ax, style="b")
