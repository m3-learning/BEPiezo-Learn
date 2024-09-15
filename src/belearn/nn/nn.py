from autophyslearn.spectroscopic.nn import Multiscale1DFitter, Model
from m3util.ml.rand import set_seeds
import itertools
from autophyslearn.postprocessing.complex import ComplexPostProcessor
from belearn.functions.sho import SHO_nn
import torch
import gc


def static_state_decorator(func):
    """Decorator that stops the function from changing the state

    Args:
        func (method): any method
    """
    def wrapper(*args, **kwargs):
        current_state = args[0].get_state
        out = func(*args, **kwargs)
        args[0].set_attributes(**current_state)
        return out
    return wrapper

def get_model_filename(basepath, results, noise, optimizer):
    """
    Get the model filename for a specific optimizer from the results dictionary.

    Parameters:
    basepath (str): The base directory path where the model file is located.
    results (dict): The dictionary containing model information.
    noise (float): The noise level for which to retrieve the model.
    optimizer (str): The optimizer type ('Adam', 'Trust Region CG', etc.).

    Returns:
    str: The full path to the model file.
    """
    return basepath + "/" + results[(noise, optimizer)]["filename"].split("//")[-1]


def instantiate_fitter(
    SHO_nn, dataset, postprocessor, input_channels=2, output_channels=4
):
    """
    Instantiate the Multiscale1DFitter model.

    Parameters:
    SHO_nn (callable): The function to fit the SHO data.
    dataset (object): The dataset object containing frequency and scaler information.
    postprocessor (callable): The function used to post-process model outputs.
    input_channels (int): Number of input channels for the model. Default is 2.
    output_channels (int): Number of output channels for the model. Default is 4.

    Returns:
    Multiscale1DFitter: An instance of the model fitter.
    """
    return Multiscale1DFitter(
        SHO_nn,  # function
        dataset.frequency_bin,  # x data
        input_channels,  # input channels
        output_channels,  # output channels
        dataset.SHO_scaler,
        postprocessor,
    )


def instantiate_model(fitter_model, dataset, noise, optimizer, training=True):
    """
    Instantiate the model with the given fitter and dataset.

    Parameters:
    fitter_model (Multiscale1DFitter): The fitter model to be used.
    dataset (object): The dataset to train the model on.
    noise (float): The noise level associated with the model.
    optimizer (str): The optimizer used to train the model.
    training (bool): If the model should be instantiated for training. Default is True.

    Returns:
    Model: The instantiated model ready for training.
    """
    return Model(
        fitter_model,
        dataset,
        training=training,
        model_basename=f"SHO_Fitter_noise_{noise}_optimizer_{optimizer}",
    )


def create_models(basepath, results, noise, dataset, postprocessor, SHO_nn):
    """
    Create and instantiate models for both Adam and Trust Region optimizers.

    Parameters:
    basepath (str): The base directory path where the model files are located.
    results (dict): Dictionary containing the trained models' results.
    noise (float): The noise level to search for the best model.
    dataset (object): The dataset to be used for the model.
    postprocessor (callable): The post-processing function.
    SHO_nn (callable): The SHO function used for the Multiscale1DFitter.

    Returns:
    tuple: A tuple containing the Adam model and the Trust Region model.
    """
    # Get filenames for both optimizers
    model_name_adam = get_model_filename(basepath, results, noise, "Adam")
    model_name_trust_region = get_model_filename(
        basepath, results, noise, "Trust Region CG"
    )

    # Instantiate the model fitters
    adam_fitter = instantiate_fitter(SHO_nn, dataset, postprocessor)
    trust_region_fitter = instantiate_fitter(SHO_nn, dataset, postprocessor)

    # Instantiate the models
    adam_model = instantiate_model(adam_fitter, dataset, noise, "Adam")
    trust_region_model = instantiate_model(
        trust_region_fitter, dataset, noise, "Trust Region CG"
    )

    adam_model.load(model_name_adam)

    trust_region_model.load(model_name_trust_region)

    return adam_model, trust_region_model


@static_state_decorator
def batch_training(dataset, optimizers, noise_list, batch_size, epochs, seed, write_CSV="Batch_Training_Noisy_Data.csv",
                   basepath=None, early_stopping_loss=None, early_stopping_count=None, early_stopping_time=None, skip=-1, **kwargs,
                   ):

    # Generate all combinations
    combinations = list(itertools.product(
        optimizers, noise_list, batch_size, epochs, seed))

    for i, training in enumerate(combinations):
        if i < skip:
            print(
                f"Skipping combination {i}: {training[0]} {training[1]} {training[2]}  {training[3]}  {training[4]}")
            continue

        optimizer = training[0]
        noise = training[1]
        batch_size = training[2]
        epochs = training[3]
        seed = training[4]

        print(f"The type is {type(training[0])}")

        if isinstance(optimizer, dict):
            optimizer_name = optimizer['name']
        else:
            optimizer_name = optimizer

        dataset.noise = noise

        set_seeds(seed=seed)

        # constructs a test train split
        X_train, X_test, y_train, y_test = dataset.test_train_split_(
            shuffle=True)

        model_name = f"SHO_{optimizer_name}_noise_{training[1]}_batch_size_{training[2]}_seed_{training[4]}"

        print(f'Working on combination: {model_name}')

        postprocessor = ComplexPostProcessor(dataset)

        model_ = Multiscale1DFitter(SHO_nn,  # function
                                    dataset.frequency_bin,  # x data
                                    2,  # input channels
                                    4,  # output channels
                                    dataset.SHO_scaler,
                                    postprocessor)

        # instantiate the model
        model = Model(model_, dataset, training=True,
                      model_basename="SHO_Fitter_original_data")

        # fits the model
        model.fit(
            X_train,
            batch_size=batch_size,
            optimizer=optimizer,
            epochs=epochs,
            write_CSV=write_CSV,
            seed=seed,
            basepath=basepath,
            early_stopping_loss=early_stopping_loss,
            early_stopping_count=early_stopping_count,
            early_stopping_time=early_stopping_time,
            i=i,
            **kwargs,
        )

        del model, X_train, X_test, y_train, y_test

        torch.cuda.empty_cache()
        clear_all_tensors()
        gc.collect()


def clear_all_tensors():
    # Get all objects in global scope
    for obj_name in dir():
        # Filter out objects that are not tensors
        if not obj_name.startswith('_'):
            obj = globals()[obj_name]
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                del obj

    # Clear PyTorch GPU cache
    torch.cuda.empty_cache()

    # Force Python's Garbage Collector to release unreferenced memory
    gc.collect()
