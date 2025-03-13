import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Type

from belearn.dataset.dataset_new import BE_Dataset
from belearn.dataset.State import State
from belearn.util.wrappers import context_manager_decorator

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import ScalarFormatter

from scipy import fftpack

from contextlib import contextmanager
import inspect

from m3util.viz.layout import (
    layout_fig,
    add_box,
    inset_connector,
    scalebar,
    imagemap,
    FigDimConverter,
    # subfigures,
    # get_axis_pos_inches,
    # draw_line_with_text,
)

from m3util.viz.arrows import (
    draw_ellipse_with_arrow,
    #DrawArrow,
   # draw_extended_arrow_indicator,
)

from m3util.viz.text import (
    add_text_to_figure,
    set_sci_notation_label,
    labelfigs,
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
                 SHO_labels: Optional[List[Dict[str, str]]] = None
                 ):
        super().__init__()
        self.dataset = dataset
        self.Printer = Printer
        self.verbose = verbose
        self.labelfigs_ = labelfigs_
        self.image_scalebar = image_scalebar
        self.color_palette = color_palette
        self.SHO_ranges = SHO_ranges
        self.SHO_labels = SHO_labels if SHO_labels is not None else [
            {"title": "Amplitude", "y_label": "Amplitude \n (Arb. U.)"},
            {"title": "Resonance Frequency", "y_label": "Resonant \n Frequency \n (Hz)"},
            {"title": "Dampening", "y_label": "Quality Factor \n (Arb. U.)"},
            {"title": "Phase", "y_label": "Phase \n (rad)"},
        ]
    
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
    
    #@State.static_dataset_decorator
    @context_manager_decorator
    def plot_twin_axis(
        self,
        ax1,
        true,
        predict=None,
        pixel=None,
        voltage_step=None,
        add_arrows=None,
        **kwargs
    ):
        # Set the attributes for the true dataset
        self.set_attributes(**true)
        
            
        # If a pixel is not provided, select a random pixel
        if pixel is None:
            pixel = np.random.randint(0, self.num_pix)
            self.pixel = pixel # JGoddy: is this necessary? 
            
        if voltage_step is None: 
            # Get the voltage step, considering the current state
            voltage_step = self.get_voltage_step(voltage_step)
            self.voltage_step = voltage_step # JGoddy: is this necessary 

        if "raw_format" in kwargs.keys():
            self.raw_format = kwargs['raw_format']
            
        # Get the raw spectral data for the selected pixel and voltage step
        #with State.temporary_state(self, **true):
        data, x = self.raw_spectra(pixel, voltage_step, frequency=True)
        
        # Get the valid parameters for the plot method
        plot_params = mlines.Line2D([], []).properties().keys()

        # Extract kwargs for ax1 and ax2 based on valid plot parameters
        
        # Remove the prefixes for ax1 and ax2 kwargs
        ax1_kwargs = {k[len('ax1_'):]: v for k, v in kwargs.items() if k[len('ax1_'):] in plot_params and k.startswith('ax1_')}
        ax1_true_kwargs = {k[len('ax1_true_'):]: v for k, v in kwargs.items() if k[len('ax1_true_'):] in plot_params and k.startswith('ax1_true_')}
        ax1_predict_kwargs = {k[len('ax1_predict_'):]: v for k, v in kwargs.items() if k[len('ax1_predict_'):] in plot_params and k.startswith('ax1_predict_')}
        ax2_kwargs = {k[len('ax2_'):]: v for k, v in kwargs.items() if k[len('ax2_'):] in plot_params and k.startswith('ax2_')}
        ax2_true_kwargs = {k[len('ax2_true_'):]: v for k, v in kwargs.items() if k[len('ax2_true_'):] in plot_params and k.startswith('ax2_true_')}
        ax2_predict_kwargs = {k[len('ax2_predict_'):]: v for k, v in kwargs.items() if k[len('ax2_predict_'):] in plot_params and k.startswith('ax2_predict_')}
        # Extract kwargs for either plot
        either_axis_kwargs_for_plotting = {k: v for k, v in kwargs.items() if k in plot_params and not k.startswith('ax1_') and not k.startswith('ax2_')}
        
        # print("***")
        # print("plot_params: ", plot_params)
        # print("****")
        # print("ax1_kwargs: ", ax1_kwargs)
        # print("ax2_kwargs: ", ax2_kwargs)
        # print("either_axis_kwargs_for_plotting: ", either_axis_kwargs_for_plotting)
        
        ax1.plot(
            x,
            data[0].flatten(),
            # color = kwargs["ax1_color"],
            # marker = kwargs["marker"],
            # label = kwargs["ax1_label"],
            **ax1_kwargs, **ax1_true_kwargs, **either_axis_kwargs_for_plotting
        )
        
        ax2 = ax1.twinx()
        ax2.plot(
            x,
            data[1].flatten(),
            # color = kwargs["color"][1],
            # marker = kwargs["marker"],
            # label = kwargs["label"][1],
            **ax2_kwargs, **ax2_true_kwargs, **either_axis_kwargs_for_plotting
        )
        
       
        
         # Ensure ax2 is drawn on top of ax1 by setting a higher zorder
        ax1.set_zorder(ax2.get_zorder() + 1)

        # Remove the axes background (set to transparent)
        ax1.set_facecolor("none")
  
        # If a predicted dataset is provided, plot its:
        # (amplitude and phase) or (real and imaginary components) etc. 
        if predict is not None:
            self.set_attributes(**predict)
            data, x = self.raw_spectra(
                pixel, voltage_step, frequency=True, **kwargs
            )
            ax1.plot(
                x, data[0].flatten(), 
                "bo", label= ax1_predict_kwargs["label"] #self.label + " " + ax1_kwargs["label"]
                #**ax1_predict_kwargs
            )
            ax2.plot(x, data[1].flatten(),
                     "ro", label= ax2_predict_kwargs["label"] #self.label + " " + ax2_kwargs["label"]
                     #**ax2_predict_kwargs
            )
            self.set_attributes(**true)

        ax1.set_xlabel(kwargs.get("x_label"))
        ax1.set_ylabel(kwargs.get("y1_label"))
        ax2.set_ylabel(kwargs.get("y2_label"))
        
        self._scientific_notation_dual(ax1, ax2)
        
       # Add the legend
       
       # Add the arrows
       
        return ax1, ax2
        
    
    #@State.static_dataset_decorator
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
        data, x = self.raw_spectra(pixel, voltage_step, frequency=True)

        # Plot amplitude and phase for the true dataset
        ax1.plot(
            x,
            data[0].flatten(),
            color=color_palette["mag"],
            marker="s",
            label= "True " + self.label + " Amplitude",
        )
        ax2 = ax1.twinx()
        ax2.plot(
            x,
            data[1].flatten(),
            color=color_palette["phase"],
            marker="s",
            label= "True " + self.label + " Phase",
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
                x, data[0].flatten(), "bo", label=self.label + " Amplitude"
            )
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
                ax1, # ax
                x, # x_data
                data[0].flatten(), # y_data
                add_arrows["mag_value"], # value
                add_arrows["width"], # width
                add_arrows["height"], # height
                axis=add_arrows.get("axis", "x"), # axis
                line_direction=add_arrows.get("line_direction", "horizontal"), # line_direction
                arrow_position=add_arrows.get("arrow_position", "top"), # arrow_position
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
    
    #@State.static_dataset_decorator
    @context_manager_decorator
    def plot_real_imaginary(
        self,
        ax1, 
        true, 
        predict=None, 
        pixel= 330, #None
        voltage_step= 87,#None,
        add_arrows=None,
        **kwargs,
    ):
        # Set the attributes for the true dataset
        self.set_attributes(**true)

        # Reset dataset state to complex format
        self.raw_format = "complex"

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
            data, x = self.raw_spectra(
                pixel, voltage_step, frequency=True, **kwargs
            )
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
    
    #@State.static_dataset_decorator
    @context_manager_decorator
    def raw_data_comparison(
        self,
        true,
        predict=None,
        filename=None,
        pixel= None,
        voltage_step = None,
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
        
        # PREVENT TRUE_STATE FROM BEING MODIFIED BY PREDICT_STATE
        
     

        # Initialize figure and axes for plotting
        fig, axs = layout_fig(2, 2, figsize=(5, 1.25))
        
        if pixel is None:
            pixel = 330
        if voltage_step is None:
            voltage_step = 87
        
        # kwargs["raw_format"] = "magnitude spectrum"
        # kwargs["ax1_color"] = color_palette["mag"] 
        # kwargs["ax2_color"] = color_palette["phase"]
        # kwargs["marker"] = "s"
        # kwargs["ax1_label"] = self.label + " Amplitude"
        # kwargs["ax2_label"] = self.label + " Phase"
        # kwargs["x_label"] = "Frequency (Hz)"
        # kwargs["y1_label"] = "Amplitude (Arb. U.)"
        # kwargs["y2_label"] = "Phase (deg)"
        ax_mag, ax_phase = self.plot_twin_axis(
            axs[0], true, predict, pixel, voltage_step, fig=fig,
            raw_format = "magnitude spectrum",
            ax1_true_color = color_palette["mag"],
            ax2_true_color = color_palette["phase"],
            ax1_predict_color = 'blue',
            ax2_predict_color = 'red',
            ax1_predict_marker = 'o',
            ax2_predict_marker = 'o',
            marker = "s",
            ax1_true_label = true["label"] + " Amplitude",
            ax2_true_label = true["label"] + " Phase",
            ax1_predict_label = predict["label"] + " Amplitude" if predict is not None else None,
            ax2_predict_label = predict["label"] + " Phase" if predict is not None else None,
            x_label = "Frequency (Hz)",
            y1_label = "Amplitude (Arb. U.)",
            y2_label = "Phase (deg)",
        )
        
        # kwargs["raw_format"] = "complex"
        
        # kwargs["ax1_color"] = color_palette["real"]
        # kwargs["ax2_color"] = color_palette["imag"]
        # kwargs["ax1_label"] = self.label + " Real"
        # kwargs["ax2_label"] = self.label + " Imag"
        # kwargs["y1_label"] = "Real (Arb. U.)"
        # kwargs["y2_label"] = "Imag (Arb. U.)"
        
        ax_real,ax_imag = self.plot_twin_axis(
            axs[1], true, predict, pixel, voltage_step, fig=fig,
            raw_format = "complex",
            ax1_true_color = color_palette["real"],
            ax2_true_color = color_palette["imag"],
            ax1_predict_color = 'blue',
            ax2_predict_color = 'red',
            ax1_predict_marker = 'o',
            ax2_predict_marker = 'o',
            ax1_true_label = true["label"] + " Real",
            ax2_true_label = true["label"] + " Imag",
            ax1_predict_label = predict["label"] + " Real" if predict is not None else None,
            ax2_predict_label = predict["label"] + " Imag" if predict is not None else None,
            y1_label = "Real (Arb. U.)",
            y2_label = "Imag (Arb. U.)",
            x_label = "Frequency (Hz)",
        )

        # ax_mag, ax_phase = self.plot_magnitude_spectrum(
        #     axs[0], true, predict, pixel, voltage_step, fig=fig, **kwargs
        # )


        # ax_real, ax_imag = self.plot_real_imaginary(
        #     axs[1], true, predict, pixel, voltage_step, **kwargs
        # )

        # Adjust the format of the tick labels and box aspect for all axes
        axes = [ax_mag, ax_real, ax_phase, ax_imag]

        for ax in axes:
            ax.set_box_aspect(1)

        # Optionally print the dataset states
        if self.verbose:
            print("True \n")
            true_state = self.set_attributes(**true)
            if predict is not None:
                print("predicted \n")
                predict_state = self.set_attributes(**predict)

        # Display the legend if requested
        if legend:
            fig.legend(bbox_to_anchor=(1.0, 1), loc="upper right", borderaxespad=0.1)

        # Save the figure if a Printer object and filename are provided
        if self.Printer is not None and filename is not None:
            self.Printer.savefig(
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
        
    
    #@State.static_dataset_decorator
    @context_manager_decorator
    def plot_hysteresis_waveform(self, fig, ax, inset_pos, x_start, x_end, y_inset_min=-2, y_inset_max=20):
        
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
        
            
    #@State.static_dataset_decorator
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
            self.be_center_frequency
            - self.be_bandwidth
            - self.be_bandwidth * 0.25,
            self.be_center_frequency
            + self.be_bandwidth
            + self.be_bandwidth * 0.25,
        )

        self.plot_hysteresis_waveform(fig, ax[2], inset_pos, x_start, x_end)

        # Set the dataset state to retrieve the magnitude spectrum
        self.scaled = False
        self.raw_format = "magnitude spectrum"
        self.measurement_state = "all"
        self.resampled = False

        # Get the magnitude spectrum for the selected pixel and voltage step
        data_ = self.raw_spectra(pixel, voltagestep)

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
        data_ = self.raw_spectra(pixel, voltagestep)

        # Plot the real and imaginary components of the spectra
        ax[4].plot(self.frequency_bin, data_[0].flatten(), label="Real")
        ax[4].set(xlabel="Frequency (Hz)", ylabel="Real (Arb. U.)")
        ax3 = ax[4].twinx()
        ax3.plot(self.frequency_bin, data_[1].flatten(), "r", label="Imaginary")
        ax3.set(xlabel="Frequency (Hz)", ylabel="Imag (Arb. U.)", facecolor="none")
        
        set_sci_notation_label(ax[1],axis="x",corner = "bottom right")
        set_sci_notation_label(ax[2],axis="x",corner = "bottom right")
        set_sci_notation_label(ax[3],axis="x",corner = "bottom right")
        set_sci_notation_label(ax[4],axis="x",corner = "bottom right")


        # Save the figure if a Printer object is available
        if self.Printer is not None:
            self.Printer.savefig(fig, filename, label_figs=ax, style="b")
            
        
    @State.static_scale_decorator
    def SHO_hist(self, SHO_data, filename=None, scaled=False):
        


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
            4 * len(SHO_data), 4, figsize=(15, 1.25 * len(SHO_data)) # figsize=(5.25, 1.25 * len(SHO_data))
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
                
                set_sci_notation_label(ax,axis="x",corner="bottom right")
                set_sci_notation_label(ax,axis="y",corner="top left")

                
                ax.xaxis.labelpad = 0 #10

                ax.set_box_aspect(1)

            if self.verbose:
                self.extraction_state

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
            pixel = np.random.randint(0, self.num_pix)
            data = self.SHO_fit_results()[[pixel], :, :]

        # Initialize the figure and axes with a 4x4 grid layout
        fig, axs = layout_fig(4, 4, figsize=(5.5, 1.1))

        # Loop over each axis and corresponding SHO label to plot the fit results
        for i, (ax, label) in enumerate(zip(axs, self.SHO_labels)):
            ax.plot(self.dc_voltage, data[0, :, i])
            ax.set_ylabel(label["y_label"])
            
            
            if abs(int('{:.1e}'.format(np.min(data[0, :, i])).split('e')[1])) > 2:
                set_sci_notation_label(ax,axis="y",corner="top left")
                
                
            if i == 3:
                ax.set_yticks([3,0,-3])
                
            ax.set_xticks([-15,0,15])
            

        # If verbose mode is enabled, log the current extraction state (for debugging or tracking)
        if self.verbose:
            self.extraction_state

        # If a Printer object is defined, save the figure with the specified filename and style
        if self.Printer is not None:
            self.Printer.savefig(fig, filename, label_figs=axs, style="b")

    
    
###### MOVIES #####

    #@State.static_dataset_decorator
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
    
    
    #@static_dataset_decorator
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

    #@static_dataset_decorator
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

        data = self.raw_spectra(pixel=pixel, voltage_step=voltage_step)

        # plot real and imaginary components of resampled data
        fig = plt.figure(figsize=(3, 1.25), layout="compressed")
        axs = plt.subplot(111)

        self.dataset.raw_format = "complex"

        data, x = self.raw_spectra(
            pixel, voltage_step, frequency=True, **kwargs
        )

        axs.plot(x, data[0].flatten(), "k", label=self.dataset.label + " Real")
        axs.set_xlabel("Frequency (Hz)")
        axs.set_ylabel("Real (Arb. U.)")
        ax2 = axs.twinx()
        ax2.set_ylabel("Imag (Arb. U.)")
        ax2.plot(x, data[1].flatten(), "g", label=self.dataset.label + " Imag")
        self._scientific_notation_dual(axs,ax2)


        axes = [axs, ax2]

        for ax in axes:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.set_box_aspect(1)

        if self.verbose:
            self.dataset.extraction_state

        if legend:
            fig.legend(bbox_to_anchor=(1.0, 1), loc="upper right", borderaxespad=0.1)

        # prints the figure
        if self.Printer is not None and filename is not None:
            self.Printer.savefig(fig, filename, style="b")
    
    
    