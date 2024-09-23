import pandas as pd
import pandas as pd


class ModelAnalysis:
    def __init__(self, DataFed=False, basepath=None, filename=None):
        self.basepath = basepath
        self.filename = filename
        self.df = pd.read_csv(basepath + "/" + filename)
        self.df["Noise Level"] = self.df["Model Name"].apply(
            lambda x: float(x.split("_")[3])
        )
        self.results = self.find_best_model()
        self.df_new = self.convert_csv_df()


def find_best_model(basepath, filename):
    """
    Finds the best model configurations based on the minimum training loss for
    each combination of noise level and optimizer in a given dataset.

    Parameters:
    basepath (str): The base directory path where the CSV file is located.
    filename (str): The name of the CSV file containing model information.

    Returns:
    dict: A dictionary where each key is a tuple (noise_level, optimizer), and
          the corresponding value is a dictionary of the row with the minimum
          training loss for that combination.
    """

    # Read the CSV file into a DataFrame
    df = pd.read_csv(basepath + "/" + filename)

    # Extract the noise level from the 'Model Name' column by splitting the string
    # assuming the noise level is the 4th part after splitting by underscores.
    df["Noise Level"] = df["Model Name"].apply(lambda x: float(x.split("_")[3]))

    # Initialize an empty dictionary to store the results
    results = {}

    # Loop over each unique combination of noise level and optimizer
    for noise_level in df["Noise Level"].unique():
        for optimizer in df["Optimizer"].unique():
            # Create a mask to filter rows for the current combination
            mask = (df["Noise Level"] == noise_level) & (df["Optimizer"] == optimizer)

            # Check if there are rows matching this combination
            if df[mask].shape[0] > 0:
                # Find the index of the row with the minimum 'Train Loss' value
                min_loss_index = df.loc[mask, "Train Loss"].idxmin()

                # Store the model configuration and details corresponding to
                # the minimum 'Train Loss' in the results dictionary
                results[(noise_level, optimizer)] = df.loc[min_loss_index].to_dict()

    # Return the dictionary of best model configurations
    return results


def convert_csv_df(df):
    """
    Converts the original dataframe to a new format with specific columns.

    Parameters:
    df (pd.DataFrame): The original dataframe.

    Returns:
    pd.DataFrame: The converted dataframe with specific columns.
    """

    # Assuming original dataframe is named `df`
    df_new = pd.DataFrame()

    # Mapping columns from the original dataframe to the new one
    df_new["batch_size"] = df["Batch Size"]
    df_new["early_stopping"] = df["Early Stoppage"]
    df_new["epoch"] = df["Epochs"]
    df_new["loss_func"] = df["Loss Function"]
    df_new["model_updates"] = df["Model Updates"]
    df_new["noise_level"] = df["Noise"]
    df_new["optimizer_name"] = df["Optimizer"]
    df_new["seed"] = df["Seed"]

    # Extracting the timestamp from the filename
    df_new["timestamp"] = pd.to_datetime(
        df["Filename"].str.extract(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")[0],
        format="%Y-%m-%d_%H-%M-%S",
    )

    df_new["total_time"] = df["Training_Time"]
    df_new["train_loss"] = df["Train Loss"]
    df_new["training_index"] = df.index

    return df_new
