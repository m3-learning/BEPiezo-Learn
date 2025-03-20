from datafed_torchflow.datafed import DataFed
from typing import Optional, Union
from pathlib import Path
from m3util.util.hashing import calculate_h5file_checksum


class BE_DataFed(DataFed):
    datafed: Optional[Union[None, str, Path]] = None
    dataset_id: Optional[str] = None


    """
    DataFed class for BE data

            dataset_id (str, optional): The DataFed ID of the dataset. Defaults to None.

    """

    def __init__(self, datafed, dataset_id, **kwargs):
        super().__init__(datafed, dataset_id = self.dataset_id, logging = True)
        self.datafed = datafed
        self.dataset_id = dataset_id
        
        self.instantiate_datafed()


def instantiate_datafed(self):
        """
        Instantiate the DataFed object and upload the dataset to DataFed if applicable.

        This method checks if the `datafed` attribute is provided and valid. If it starts with "d/",
        it is assumed to be a DataFed ID. Otherwise, it attempts to create a DataFed object and upload
        the dataset to DataFed, extracting metadata and handling the upload process.

        Raises:
            ValueError: If the `datafed` attribute is not a valid DataFed path or identifier.
        """
        if self.datafed is not None and self.datafed.startswith("d/"):
            # If datafed is a DataFed ID, set it directly
            self.dataset_id = self.datafed
        elif self.datafed is not None:
            # Instantiate the DataFed object if datafed is a path
            self.datafed_obj = DataFed(self.datafed, dataset_id_or_path = self.dataset_id, logging = True)

            # Extract metadata from the HDF5 file structure
            metadata = self.extract_h5_structure()

            metadata.update(
                {
                    "checksum": calculate_h5file_checksum(self.file),
                }
            )

            self.dataset_id = self.datafed_obj.upload_dataset_to_DataFed()

            # # Upload the file to DataFed
            # self.datafed_obj.upload_file(dc_resp[0].data[0].id, self.file, wait=False)


            # # Set the DataFed ID from the response
            # self.dataset_id = dc_resp[0].data[0].id

            # # Upload the file to DataFed
            # self.datafed_obj.upload_file(self.dataset_id , self.file, wait=False)

        

        elif self.datafed is None:
            # If datafed is None, set dataset_id to None
            self.dataset_id = None
        else:
            # Raise an error if datafed is not a valid path or identifier
            raise ValueError("DataFed value is not a valid DataFed path or identifier")

   