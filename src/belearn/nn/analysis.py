import pandas as pd


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
