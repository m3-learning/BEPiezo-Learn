from datafed_torchflow.pytorch import InferenceEvaluation
from autophyslearn.spectroscopic.nn import Multiscale1DFitter, Model
from autophyslearn.postprocessing.complex import ComplexPostProcessor
from belearn.functions.sho import SHO_nn
import torch.nn.functional as F


class BEInference(InferenceEvaluation):
    def __init__(self, dataframe, dataset, df_api, model=None, root_directory='./', save_directory="./tmp/", **Kwargs):
        self.dataset = dataset
                
        if model is None:
            self.build_model()
        
        # Pass all the required arguments to the superclass
        super().__init__(dataframe, 
                 dataset, 
                 df_api,
                 root_directory=root_directory, 
                 save_directory=save_directory, 
                 **Kwargs)
        
        self.get_reference_data()
            
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
    
    def get_reference_data(self, noise_state=0):
        self.dataset.noise = noise_state

        # extracts the x and y data based on the noise
        self.x_reference, self.y_reference = self.dataset.NN_data()
        
    def _getFileName(self, row):
        
        # manual implementation of the getFileName method
        return self.df_api.getFileName(row.id)[:-8]
    
    def evaluate(self, row, file_path):
        print(f'Evaluating the model: {row.id}\n from file: {file_path}')
        
        self.dataset.noise = int(row['noise_level'])

        # extracts the x and y data based on the noise
        x_data, y_data = self.dataset.NN_data()
            
        pred_data, scaled_param, parm = self.model.predict(x_data)
        
        noise_mse = F.mse_loss(x_data, self.x_reference, reduction='mean')
        mse_reference = F.mse_loss(pred_data, self.x_reference, reduction='mean')
        
        # returns dictionary of metrics.
        return {"mse_reference" : mse_reference.item(), "Noise MSE" : noise_mse.item()}
    
    