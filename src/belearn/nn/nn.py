from autophyslearn.spectroscopic.nn import Multiscale1DFitter, Model
from m3util.ml.rand import set_seeds
from m3util.pandas.filter import find_min_max_by_group
import itertools
from autophyslearn.postprocessing.complex import ComplexPostProcessor
from belearn.functions.sho import SHO_nn
import torch
import gc
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import itertools
import torch
import gc
import os


def static_state_decorator(func):
    """Decorator that stops the function from changing the state

    Args:
        func (method): any method
    """

    def wrapper(*args, **kwargs):
        current_state = args[1].get_state
        out = func(*args, **kwargs)
        args[1].set_attributes(**current_state)
        return out

    return wrapper


# def get_model_filename(basepath, results, noise, optimizer):
#     """
#     Get the model filename for a specific optimizer from the results dictionary.

#     Parameters:
#     basepath (str): The base directory path where the model file is located.
#     results (dict): The dictionary containing model information.
#     noise (float): The noise level for which to retrieve the model.
#     optimizer (str): The optimizer type ('Adam', 'Trust Region CG', etc.).

#     Returns:
#     str: The full path to the model file.
#     """
#     return basepath + "/" + results[(noise, optimizer)]["filename"].split("//")[-1]


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


def get_model(
    df,
    optimizer_name="Adam",
    noise=0,
    optimized_result="train_loss",
    find="min",
    exclude_kwargs={"early_stopping": True},
    file_name=r".*\.pth$",
    **kwargs,
):
    """
    Retrieve the model from the dataframe based on optimization results.

    Parameters:
    df (pd.DataFrame): The dataframe containing model information.
    noise (float): The noise level for which to retrieve the model. Default is 0.
    optimized_result (str): The column name in the dataframe to optimize. Default is "train_loss".
    find (str): Whether to find the "min" or "max" value in the optimized_result column. Default is "min".
    exclude_kwargs (dict): Dictionary of keyword arguments to exclude from the search. Default is {"early_stopping": True}.
    file_name (str): Regex pattern to match the model file name. Default is r".*\.pth$".
    **kwargs: Additional keyword arguments to pass to the find_min_max_by_group function.

    Returns:
    dict: The model information dictionary.
    """

    # Find the model in the dataframe based on the specified criteria
    model = find_min_max_by_group(
        df=df,
        optimized_result=optimized_result,
        find=find,
        exclude_kwargs=exclude_kwargs,
        noise_level=noise,
        optimizer_name=optimizer_name,
        file_name=file_name,
        **kwargs,
    )

    return model


def create_models(
    basepath,
    df,
    dataset,
    noise=0,
    optimized_result="train_loss",
    find="min",
    postprocessor=None,
    function=SHO_nn,
    exclude_kwargs={},
    file_name=r".*\.pth$",
    **kwargs,
):
    # Remove any keys from kwargs that are explicitly passed in get_model
    filtered_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        not in {
            "col_name",
            "find",
            "exclude_kwargs",
            "noise_level",
            "optimizer_name",
            "file_name",
        }
    }

    adam_df = get_model(
        df,
        optimized_result=optimized_result,
        find=find,
        exclude_kwargs=exclude_kwargs,
        noise=noise,
        optimizer_name="Adam",
        file_name=file_name,
        **filtered_kwargs,
    )

    TR_df = get_model(
        df,
        optimized_result=optimized_result,
        find=find,
        exclude_kwargs=exclude_kwargs,
        noise=noise,
        optimizer_name="Trust Region CG",
        file_name=file_name,
        **filtered_kwargs,
    )

    adam_path = os.path.join(basepath, adam_df["file_name"])
    TR_path = os.path.join(basepath, TR_df["file_name"])

    # Instantiate the model fitters
    adam_fitter = instantiate_fitter(function, dataset, postprocessor)
    trust_region_fitter = instantiate_fitter(function, dataset, postprocessor)

    # Instantiate the models
    adam_model = instantiate_model(adam_fitter, dataset, noise, "Adam")
    trust_region_model = instantiate_model(
        trust_region_fitter, dataset, noise, "Trust Region CG"
    )

    adam_model.load(adam_path)
    trust_region_model.load(TR_path)

    return adam_model, trust_region_model, adam_df, TR_df


@dataclass
class BatchTrainer:
    dataset: Any
    optimizers: List[Any]
    noise_list: List[float]
    batch_size: List[int]
    epochs: List[int]
    seed: List[int]
    write_CSV: str = "Batch_Training_Noisy_Data.csv"
    basepath: Optional[str] = None
    early_stopping_loss: Optional[float] = None
    early_stopping_count: Optional[int] = None
    early_stopping_time: Optional[float] = None
    skip: int = -1
    datafed_path: Optional[str] = None
    script_path: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    combinations: List[Any] = field(init=False)

    def __post_init__(self):
        """
        Post-initialization method to generate the combinations of parameters.
        """
        self.combinations = list(
            itertools.product(
                self.optimizers,
                self.noise_list,
                self.batch_size,
                self.epochs,
                self.seed,
            )
        )

    def run_training(self, dataset_obj, **kwargs):
        """
        Runs the batch training process based on the initialized combinations.
        """
        for i, training in enumerate(self.combinations):
            if i < self.skip:
                print(
                    f"Skipping combination {i}: {training[0]} {training[1]} {training[2]}  {training[3]}  {training[4]}"
                )
                continue

            optimizer, noise, batch_size, epochs, seed = training

            if isinstance(optimizer, dict):
                optimizer_name = optimizer["name"]
            else:
                optimizer_name = optimizer

            self.dataset.noise = noise

            set_seeds(seed=seed)

            # constructs a test-train split
            X_train, X_test, y_train, y_test = self.dataset.test_train_split_(
                shuffle=True
            )

            model_name = f"SHO_{optimizer_name}_noise_{noise}_batch_size_{batch_size}_seed_{seed}"
            print(f"Working on combination: {model_name}")

            postprocessor = ComplexPostProcessor(self.dataset)

            model_ = Multiscale1DFitter(
                SHO_nn,  # function
                self.dataset.frequency_bin,  # x data
                2,  # input channels
                4,  # output channels
                self.dataset.SHO_scaler,
                postprocessor,
            )

            # instantiate the model
            model = Model(
                model_,
                self.dataset,
                training=True,
                model_basename="SHO_Fitter",
                datafed_path=self.datafed_path,
                script_path=self.script_path,
                dataset_id=self.dataset.dataset_id,
            )

            # fits the model
            model.fit(
                X_train,
                batch_size=batch_size,
                optimizer=optimizer,
                epochs=epochs,
                write_CSV=self.write_CSV,
                seed=seed,
                basepath=self.basepath,
                early_stopping_loss=self.early_stopping_loss,
                early_stopping_count=self.early_stopping_count,
                early_stopping_time=self.early_stopping_time,
                i=i,
                **self.kwargs,
            )

            # Update script path if necessary
            
            # JGoddy commented out the below lines because 
            # script_path should stay as the local path so it is accessible
            # for the checksum generator 
            # if self.datafed_path is not None: 
            #     self.script_path = model.script_path

            del model, X_train, X_test, y_train, y_test

            torch.cuda.empty_cache()
            clear_all_tensors()
            gc.collect()


# @static_state_decorator
# def batch_training(
#     dataset,
#     optimizers,
#     noise_list,
#     batch_size,
#     epochs,
#     seed,
#     write_CSV="Batch_Training_Noisy_Data.csv",
#     basepath=None,
#     early_stopping_loss=None,
#     early_stopping_count=None,
#     early_stopping_time=None,
#     skip=-1,
#     datafed_path=None,
#     script_path=None,
#     **kwargs,
# ):
#     # Generate all combinations
#     combinations = list(
#         itertools.product(optimizers, noise_list, batch_size, epochs, seed)
#     )

#     for i, training in enumerate(combinations):
#         if i < skip:
#             print(
#                 f"Skipping combination {i}: {training[0]} {training[1]} {training[2]}  {training[3]}  {training[4]}"
#             )
#             continue

#         optimizer = training[0]
#         noise = training[1]
#         batch_size = training[2]
#         epochs = training[3]
#         seed = training[4]

#         print(f"The type is {type(training[0])}")

#         if isinstance(optimizer, dict):
#             optimizer_name = optimizer["name"]
#         else:
#             optimizer_name = optimizer

#         dataset.noise = noise

#         set_seeds(seed=seed)

#         # constructs a test train split
#         X_train, X_test, y_train, y_test = dataset.test_train_split_(shuffle=True)

#         model_name = f"SHO_{optimizer_name}_noise_{training[1]}_batch_size_{training[2]}_seed_{training[4]}"

#         print(f"Working on combination: {model_name}")

#         postprocessor = ComplexPostProcessor(dataset)

#         model_ = Multiscale1DFitter(
#             SHO_nn,  # function
#             dataset.frequency_bin,  # x data
#             2,  # input channels
#             4,  # output channels
#             dataset.SHO_scaler,
#             postprocessor,
#         )

#         # instantiate the model
#         model = Model(
#             model_,
#             dataset,
#             training=True,
#             model_basename="SHO_Fitter_original_data",
#             datafed_path=datafed_path,
#             script_path=script_path,
#         )

#         # fits the model
#         model.fit(
#             X_train,
#             batch_size=batch_size,
#             optimizer=optimizer,
#             epochs=epochs,
#             write_CSV=write_CSV,
#             seed=seed,
#             basepath=basepath,
#             early_stopping_loss=early_stopping_loss,
#             early_stopping_count=early_stopping_count,
#             early_stopping_time=early_stopping_time,
#             i=i,
#             **kwargs,
#         )

#         if datafed_path is not None:
#             script_path = model.script_path

#         del model, X_train, X_test, y_train, y_test

#         torch.cuda.empty_cache()
#         clear_all_tensors()
#         gc.collect()


def clear_all_tensors():
    # Get all objects in global scope
    for obj_name in dir():
        # Filter out objects that are not tensors
        if not obj_name.startswith("_"):
            obj = globals()[obj_name]
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                del obj

    # Clear PyTorch GPU cache
    torch.cuda.empty_cache()

    # Force Python's Garbage Collector to release unreferenced memory
    gc.collect()
