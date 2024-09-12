from m3util.util.IO import download_and_unzip
from belearn.dataset.dataset import BE_Dataset
import numpy as np

def test_notebook_execution():

    # Download the data file from Zenodo
    url = 'https://zenodo.org/record/7774788/files/PZT_2080_raw_data.h5?download=1'

    # Specify the filename and the path to save the file
    filename = '/data_raw.h5'
    save_path = './Data'

    # download the file
    download_and_unzip(filename, url, save_path)

    data_path = save_path + '/' + filename

    # instantiate the dataset object
    dataset = BE_Dataset(data_path)

    # print the contents of the file
    dataset.print_be_tree()

    # calculates the standard deviation and uses that for the noise
    noise_STD = np.std(dataset.get_original_data)

    # prints the standard deviation
    print(noise_STD)


    dataset.generate_noisy_data_records(noise_levels = [1], 
                                        verbose=True, 
                                        noise_STD=noise_STD)

    i = 1
    out = [f"Noisy_Data_{i}"]
    out.append("Raw_Data")

    for data in out:
        print(f"Fitting {data}")
        dataset.SHO_Fitter(dataset = data, h5_sho_targ_grp = f"{data}_SHO_Fit", max_mem=1024*64, max_cores= 20)


    # print the contents of the file
    dataset.print_be_tree()

