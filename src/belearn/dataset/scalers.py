from m3util.ml.preprocessor import GlobalScaler
from belearn.dataset.transformers import is_complex, to_complex
import numpy as np
class Raw_Data_Scaler:
        """
        Raw_Data_Scaler class that defines the scaler for band excitation data

        """

        def __init__(self,  raw_data):
            """
            __init__ Initialization function

            Args:
                raw_data (np.array): raw band excitation data to scale
            """

            self.raw_data = raw_data

            # conduct the fit on initialization
            self.fit()

        def complex_data_converter(self, data):
            """
            complex_data_converter converter that converts the dataset to complex

            Args:
                data (np.array): band excitation data to convert

            Returns:
                np.array: band excitation data as a complex number
            """
            if is_complex(data):
                return data
            else:
                return to_complex(data)

        def fit(self):
            """
            fit function to fit the scaler
            """

            # gets the raw data
            data = self.raw_data
            data = self.complex_data_converter(data)

            # extracts the real and imaginary components
            real = np.real(data)
            imag = np.imag(data)

            # does a global scaler on the data
            self.real_scaler = GlobalScaler()
            self.imag_scaler = GlobalScaler()

            # computes global scaler on the real and imaginary parts
            self.real_scaler.fit(real)
            self.imag_scaler.fit(imag)

        def transform(self, data):
            """
            transform Function to transform the data

            Args:
                data (np.array): band excitation data

            Returns:
                np.array: scaled band excitation data_
            """

            # converts the data to a complex number
            data = self.complex_data_converter(data)

            # extracts the real and imaginary components
            real = np.real(data)
            imag = np.imag(data)

            # computes the transform
            real = self.real_scaler.transform(real)
            imag = self.imag_scaler.transform(imag)

            # returns the complex number
            return real + 1j*imag

        def inverse_transform(self, data):
            """
            inverse_transform Computes the inverse transform

            Args:
                data (np.array): band excitation data

            Returns:
                np.array: unscaled band excitation data
            """

            # converts the data to complex
            data = self.complex_data_converter(data)

            # extracts the real and imaginary components
            real = np.real(data)
            imag = np.imag(data)

            # computes the inverse transform
            real = self.real_scaler.inverse_transform(real)
            imag = self.imag_scaler.inverse_transform(imag)

            return real + 1j*imag

# from m3util.converters.complex import to_complex
# from m3util.ml.preprocessor import GlobalScaler
# import numpy as np

# class Raw_Data_Scaler():
#         """
#         Raw_Data_Scaler class that defines the scaler for band excitation data

#         """

#         def __init__(self, raw_data):
#             """
#             __init__ Initialization function

#             Args:
#                 raw_data (np.array): raw band excitation data to scale
#             """

#             self.raw_data = raw_data

#             # conduct the fit on initialization
#             self.fit()

#         # @staticmethod
#         # def complex_data_converter(data):
#         #     """
#         #     complex_data_converter converter that converts the dataset to complex

#         #     Args:
#         #         data (np.array): band excitation data to convert

#         #     Returns:
#         #         np.array: band excitation data as a complex number
#         #     """
#         #     if to_complex(data):
#         #         return data
#         #     else:
#         #         return to_complex(data)

#         def fit(self):
#             """
#             fit function to fit the scaler
#             """

#             # gets the raw data
#             data = self.raw_data
#             data = to_complex(data)  #TODO removed to_complex

#             # extracts the real and imaginary components
#             real = np.real(data)
#             imag = np.imag(data)

#             # does a global scaler on the data
#             self.real_scaler = GlobalScaler()
#             self.imag_scaler = GlobalScaler()

#             # computes global scaler on the real and imaginary parts
#             self.real_scaler.fit(real)
#             self.imag_scaler.fit(imag)

#         def transform(self, data):
#             """
#             transform Function to transform the data

#             Args:
#                 data (np.array): band excitation data

#             Returns:
#                 np.array: scaled band excitation data_
#             """

#             # converts the data to a complex number
#             data = to_complex(data)

#             # extracts the real and imaginary components
#             real = np.real(data)
#             imag = np.imag(data)

#             # computes the transform
#             real = self.real_scaler.transform(real)
#             imag = self.imag_scaler.transform(imag)

#             # returns the complex number
#             return real + 1j*imag

#         def inverse_transform(self, data):
#             """
#             inverse_transform Computes the inverse transform

#             Args:
#                 data (np.array): band excitation data

#             Returns:
#                 np.array: unscaled band excitation data
#             """

#             # converts the data to complex
#             data = to_complex(data)

#             # extracts the real and imaginary componets
#             real = np.real(data)
#             imag = np.imag(data)

#             # computes the inverse transform
#             real = self.real_scaler.inverse_transform(real)
#             imag = self.imag_scaler.inverse_transform(imag)

#             return real + 1j*imag