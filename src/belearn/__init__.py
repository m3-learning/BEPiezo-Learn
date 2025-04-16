import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "BEPiezo-Learn"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
from belearn import dataset
from belearn import filters
from belearn import functions
from belearn import nn
from belearn import util

from belearn.dataset import (BE_DataFed, BE_Dataset, BE_model_utils, Datafed,
                             MSE, Preprocessing, Raw_Data_Scaler, State,
                             analytics, dataset, dataset_new, get_rankings,
                             instantiate_datafed, is_complex, model_utils,
                             mse_rankings, preprocessing, print_mse, scalers,
                             to_complex, to_real_imag, transformers,)
from belearn.filters import (clean_interpolate, filters,)
from belearn.functions import (SHO_nn, hysteresis, hysteresis_nn, sho,)
from belearn.nn import (BEInference, BatchTrainer, ModelAnalysis, analysis,
                        clear_all_tensors, convert_csv_df, create_models,
                        find_best_model, get_model, inference,
                        instantiate_fitter, instantiate_model, nn,
                        static_state_decorator,)
from belearn.util import (context_manager_decorator, static_state_decorator,
                          temporary_state, wrappers,)

__all__ = ['BEInference', 'BE_DataFed', 'BE_Dataset', 'BE_model_utils',
           'BatchTrainer', 'Datafed', 'MSE', 'ModelAnalysis', 'Preprocessing',
           'Raw_Data_Scaler', 'SHO_nn', 'State', 'analysis', 'analytics',
           'clean_interpolate', 'clear_all_tensors',
           'context_manager_decorator', 'convert_csv_df', 'create_models',
           'dataset', 'dataset_new', 'filters', 'find_best_model', 'functions',
           'get_model', 'get_rankings', 'hysteresis', 'hysteresis_nn',
           'inference', 'instantiate_datafed', 'instantiate_fitter',
           'instantiate_model', 'is_complex', 'model_utils', 'mse_rankings',
           'nn', 'preprocessing', 'print_mse', 'scalers', 'sho',
           'static_state_decorator', 'temporary_state', 'to_complex',
           'to_real_imag', 'transformers', 'util', 'wrappers']
