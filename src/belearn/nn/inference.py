from datafed_torchflow.pytorch import InferenceEvaluation
from autophyslearn.spectroscopic.nn import Multiscale1DFitter, Model
from autophyslearn.postprocessing.complex import ComplexPostProcessor
from belearn.functions.sho import SHO_nn

class BEInference(InferenceEvaluation):
    def __init__(self, dataframe, dataset, model=None, df_api=None, root_directory=None, save_directory=None):
        # Pass all the required arguments to the superclass
        super().__init__(dataframe, dataset, model, df_api, root_directory, save_directory)
        
        if model is None:
            self.build_model()
            
    def build_model(self):
        postprocessor = ComplexPostProcessor(self.dataset)


        model_ = Multiscale1DFitter(SHO_nn,  # function
                                    self.dataset.frequency_bin,  # x data
                                    2,  # input channels
                                    4,  # output channels
                                    self.dataset.SHO_scaler,
                                    postprocessor)
        
        # instantiate the model
        model = Model(model_, self.dataset, training=False)
        
        return model
    
    def run(self):
        for i, row in self.df.iterrows():
            filename = self.df_api.getFileName(row.id) 