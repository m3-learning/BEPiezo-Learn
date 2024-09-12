import numpy as np
from m3util.viz.printing import printer
from m3util.viz.style import set_style
from m3util.ml.rand import set_seeds
from m3util.util.IO import download_and_unzip
from belearn.dataset.dataset import BE_Dataset
from belearn.viz.viz import Viz
from belearn.functions.sho import SHO_nn
from belearn.dataset.analytics import print_mse
from autophyslearn.postprocessing.complex import ComplexPostProcessor
from autophyslearn.spectroscopic.nn import Multiscale1DFitter, Model

def test_notebook_execution():
    printing = printer(basepath = './Figures/')


    set_style("printing")
    set_seeds(seed=42)

    # Specify the filename and the path to save the file
    filename = 'data_raw.h5'
    save_path = "./Data"

    data_path = save_path + "/" + filename


    # instantiate the dataset object
    dataset = BE_Dataset(data_path, SHO_fit_func_LSQF=SHO_nn)

    # print the contents of the file
    dataset.print_be_tree()


    # insatiate the visualization object
    image_scalebar = [2000, 500, "nm", "br"]

    BE_viz = Viz(dataset, printing, verbose=True, 
                SHO_ranges = [(0,1.5e-4), (1.31e6, 1.33e6), (-300, 0), (-np.pi, np.pi)],
                image_scalebar=image_scalebar)

    true = {"resampled": False,
            "label": "Raw",
            "noise": 0, 
            "measurement_state": "all"}

    predicted = {"fitter": "LSQF", "resampled": False, "label": "Raw", "scaled": False, "noise" : 0}

    BE_viz.fit_tester(true, predicted, filename="Figure_7_PyTorch_fit_tester")



    state = {"fitter": "LSQF", "resampled": True, "scaled": True, "label": "Scaled"}

    BE_viz.nn_checker(state, filename="Figure_8_Scaled Raw Data")


    dataset.LSQF_phase_shift = np.pi / 2

    BE_viz.SHO_hist(
        dataset.SHO_fit_results(), filename="Figure_9_Phase_Shifted_Scaled_Histograms"
    )


    set_seeds(seed=42)

    postprocessor = ComplexPostProcessor(dataset)

    model_ = Multiscale1DFitter(SHO_nn, # function 
                                dataset.frequency_bin, # x data
                                2, # input channels
                                4, # output channels
                                dataset.SHO_scaler, 
                                postprocessor)

    # instantiate the model
    model = Model(model_, dataset, training=True, model_basename="SHO_Fitter_original_data")


    # constructs a test train split
    X_train, X_test, y_train, y_test = dataset.test_train_split_(shuffle=True)

    train = True

    if train:
        # fits the model
        model.fit(
            dataset.X_train,
            500,
            optimizer="Adam",
            epochs = 5,
        )
    else:
        model.load(
            "/home/ferroelectric/m3_learning/m3_learning/papers/2023_Rapid_Fitting/Trained Models/SHO Fitter/SHO_Fitter_original_data_model_epoch_5_train_loss_0.0449272525189978.pth"
        )


    d1, d2, index1, mse1 = BE_viz.bmw_nn(
        X_train,
        prediction=model,
        out_state={"scaled": True, "raw_format": "complex"},
        returns=True,
        filename="Figure_10_NN_validation_Train",
    )


    d1, d2, index1, mse1 = BE_viz.bmw_nn(
        X_test,
        prediction=model,
        out_state={"scaled": True, "measurement State": "complex"},
        returns=True,
        filename="Figure_11_NN_validation_test",
    )

    state = {
        "fitter": "LSQF",
        "raw_format": "complex",
        "resampled": False,
        "scaled": True,
        "output_shape": "index",
    }

    X_data, Y_data = dataset.NN_data()

    d1, d2, index1, mse1 = BE_viz.bmw_nn(
        state,
        prediction=model,
        out_state={"scaled": True, "measurement State": "complex"},
        returns=True,
        filename="Figure_12_NN_validation_full_data",
    )


    X_data, Y_data = dataset.NN_data()
    LSQF_ = {'resampled': True,
                    'raw_format': 'complex',
                    'fitter': 'LSQF',
                    'scaled': True,
                    'output_shape': 'index',
                    'measurement_state': 'all',
                    'resampled_bins': 165,
                    'LSQF_phase_shift': 1.5707963267948966,
                    'NN_phase_shift': None,
                    'noise': 0}

    data = (LSQF_, X_data, X_test, X_train)
    labels = ["LSQF", "Full Data", "Test Data", "Train Data"]

    print_mse(model_, model, data, labels)



    X_data, Y_data = dataset.NN_data()

    model.inference_timer(X_data, batch_size=1000)


    # we will add the appropriate phase shift to the dataset based on the fix seed,
    # If your seed is different the results might vary
    dataset.NN_phase_shift = np.pi/2 

    # you can view the test and training dataset by replacing X_data with X_test or X_train
    pred_data, scaled_param, parm = model.predict(X_data)

    BE_viz.SHO_hist(parm, filename="Figure_13_NN_Unscaled_Parameters_Histograms")




    out = dataset.SHO_scaler.transform(parm)

    BE_viz.SHO_hist(out, filename="Figure_13_NN_scaled_Parameters_Histograms", scaled=True)





    # you can view the test and training dataset by replacing X_data with X_test or X_train
    pred_data, scaled_param, parm = model.predict(X_data)

    BE_viz.SHO_switching_maps(parm, filename="Figure_15_NN_Switching_Maps")



    # Insatiate the visualization object
    image_scalebar = [2000, 500, "nm", "br"]

    BE_viz = Viz(dataset, printing, verbose=True, 
                SHO_ranges = [(0,1.5e-4), (1.31e6, 1.33e6), (-300, 0), (-np.pi, np.pi)],
                image_scalebar=image_scalebar)



    BE_viz.SHO_fit_movie_images(noise = 0, 
                                models = [model],
                                scalebar_= True, 
                                basepath = "Movies/SHO_NN_",  
                                filename="SHO_NN",
                                phase_shift = [np.pi/2],)


    # ## Comparison SHO and Neural Network Fits
    # 
    # 3 graphs, best, median, worst
    # 
    # histograms of parameters.
    # 



    # sets the phase shift of the dataset
    dataset.NN_phase_shift = np.pi/2
    dataset.LSQF_phase_shift = np.pi/2
    dataset.measurement_state = "all"

    # sets the true state which to compare the results.
    true_state = {
        "fitter": "LSQF",
        "raw_format": "complex",
        "resampled": True,
        "scaled": True,
        "output_shape": "index",
        "measurement_state": "all",
    }

    # sets the state of the output data
    out_state = {"scaled": True, "raw_format": "magnitude spectrum"}

    # sets the number of examples to get
    n = 1

    LSQF = BE_viz.get_best_median_worst(
        true_state,
        prediction={"fitter": "LSQF"},
        out_state=out_state,
        SHO_results=True,
        n=n,
    )

    NN = BE_viz.get_best_median_worst(
        true_state, prediction=model, out_state=out_state, SHO_results=True, n=n
    )

    data = (LSQF, NN)
    names = ["LSQF", "NN"]

    BE_viz.SHO_Fit_comparison(
        data,
        names,
        model_comparison=[model, {"fitter": "LSQF"}],
        out_state=out_state,
        filename="Figure_14_LSQF_NN_bmw_comparison",
        verbose=False,
    )



    true_state = {
        "fitter": "LSQF",
        "raw_format": "complex",
        "resampled": True,
        "scaled": True,
        "output_shape": "index",
        "measurement_state": "all",
    }


    BE_viz.violin_plot_comparison_SHO(true_state, model, X_data, filename="Figure_16_Violin") 



    BE_viz.SHO_fit_movie_images(noise = 0, 
                                models=[None, model], 
                                scalebar_= True, 
                                basepath = "Movies/SHO_NN_LSQF_Compare_",  
                                filename="SHO_NN_LSQF_Compare", 
                                labels = ['LSQF', 'NN'],
                                phase_shift = [np.pi/2, np.pi/2])



    dataset.measurement_state = "all"
    dataset.NN_phase_shift = np.pi/2
    dataset.LSQF_phase_shift = np.pi/2

    true_state = {
        "fitter": "LSQF",
        "raw_format": "complex",
        "resampled": True,
        "scaled": True,
        "output_shape": "index",
        "measurement_state": "all",
    }

    # gets the parameters from the SHO LSQF fit
    true = dataset.SHO_fit_results(state = true_state).reshape(-1, 4)

    # finds the index less than a certain value
    ind = np.argwhere(true[:,1]< -3).flatten()


    true_state = {
        "fitter": "LSQF",
        "raw_format": "complex",
        "resampled": True,
        "scaled": True,
        "output_shape": "index",
        "measurement_state": "all"
    }

    out_state = {"raw_format": "magnitude spectrum", "measurement_state": "all"}

    n = 1

    LSQF = BE_viz.get_best_median_worst(
        true_state,
        prediction={"fitter": "LSQF"},
        out_state=out_state,
        SHO_results=True,
        n=n,
        index = ind,
        verbose = False
    )

    NN = BE_viz.get_best_median_worst(
        true_state, prediction=model,
        out_state=out_state, SHO_results=True, n=n, index = ind, verbose = False,
    )

    data = (LSQF, NN)
    names = ["LSQF", "NN"]

    BE_viz.SHO_Fit_comparison(
        data,
        names,
        model_comparison=[model, {"fitter": "LSQF"}],
        out_state=out_state,
        filename="Figure_14_LSQF_NN_bmw_comparison",
        verbose = False
    )





