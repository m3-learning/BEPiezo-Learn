from belearn.dataset.fitters.base import BaseFitter
from belearn.util.wrappers import context_manager_decorator
from typing import Optional, Any
import h5py
import numpy as np
import sidpy
from pyUSID.io.hdf_utils import reshape_to_n_dims, get_auxiliary_datasets
from belearn.filters.filters import clean_interpolate


class HysteresisFitter(BaseFitter):
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
