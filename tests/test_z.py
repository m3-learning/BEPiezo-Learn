import os
import shutil


def cleanup_directories():
    """
    Fixture that deletes the 'Data' and 'Figures' directories after the test session.
    This runs once after all the tests are complete.
    """
    yield  # Run tests first

    # Define the directories to be deleted
    directories = ["Data", "Figures", "Movies"]

    # Loop through each directory and delete it if it exists
    for directory in directories:
        if os.path.exists(directory):
            print(f"Deleting directory: {directory}")
            shutil.rmtree(directory)  # Recursively delete the directory
        else:
            print(f"Directory not found: {directory}")