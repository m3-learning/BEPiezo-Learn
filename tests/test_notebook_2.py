from m3util.viz.printing import printer
from m3util.viz.style import set_style
from m3util.ml.rand import set_seeds
from m3util.util.IO import download_and_unzip
from belearn.dataset.dataset import BE_Dataset
from belearn.viz.viz import Viz
import numpy as np


def test_notebook_execution():
    # from m3_learning.be.dataset import BE_Dataset
    printing = printer(basepath="./Figures/")

    set_style("printing")
    set_seeds(seed=42)


    # Download the data file from Zenodo
    url = "https://zenodo.org/record/7774788/files/PZT_2080_raw_data.h5?download=1"

    # Specify the filename and the path to save the file
    filename = "data_raw.h5"
    save_path = "./Data"

    # download the file
    download_and_unzip(filename, url, save_path)


    data_path = save_path + "/" + filename

    # instantiate the dataset object
    dataset = BE_Dataset(data_path)

    # print the contents of the file
    dataset.print_be_tree()


    # insatiate the visualization object
    image_scalebar = [2000, 500, "nm", "br"]

    BE_viz = Viz(
        dataset,
        printing,
        verbose=True,
        SHO_ranges=[(0, 1.5e-4), (1.31e6, 1.33e6), (-300, 300), (-np.pi, np.pi)],
        image_scalebar=image_scalebar,
    )

    prediction = {"resampled": False, "label": "Raw"}

    BE_viz.raw_data_comparison(prediction, filename="Figure_1_raw_cantilever_response")


    BE_viz.raw_be(dataset, filename="Figure_2_raw_be_experiment")


    Fit_SHO = False

    if Fit_SHO:

        # # computes the SHO fit for the data in the file
        dataset.SHO_Fitter(force=True)

        # instantiate the dataset object
        # good to instantiate the dataset object after fitting
        dataset = BE_Dataset(data_path)


    dataset.LSQF_phase_shift = 0

    BE_viz.SHO_hist(dataset.SHO_fit_results(), filename="Figure_3_Original_LSQF_Histograms")


    dataset.LSQF_phase_shift = np.pi / 2

    BE_viz.SHO_hist(
        dataset.SHO_fit_results(), filename="Figure_4_Phase_Shifted_LSQF_Histograms"
    )


    BE_viz.dataset.measurement_state = "on"

    BE_viz.SHO_loops(filename="Figure_5_Single_Pixel_Loops")


    BE_viz.SHO_fit_movie_images(
        noise=0,
        scalebar_=True,
        basepath="Movies/SHO_LSQF_",
        filename="SHO_LSQF",
        phase_shift=[np.pi / 2],
    )
