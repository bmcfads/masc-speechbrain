import logging
import os
import shutil

from speechbrain.utils.data_utils import download_file

try:
    import pandas as pd
except ImportError:
    err_msg = (
        "The optional dependency pandas must be installed to run this recipe.\n"
    )
    err_msg += "Install using `pip install pandas`.\n"
    raise ImportError(err_msg)

logger = logging.getLogger(__name__)

def prepare_STOP(data_folder, save_folder, type, domains=[], flat_intents=False, skip_prep=False):
    """
    This function prepares the STOP dataset.
    If the folder does not exist, the zip file will be extracted. If the zip file does not exist, it will be downloaded.

    data_folder : path to STOP dataset.
    save_folder: path where to save the csv manifest files.
    slu_type : one of the following:

      "direct":{input=audio, output=semantics}

    domains : list of domain to include; if empty, all domains included.
    flat_intents : if True, exclude nested intents and only use flat intents.
    skip_prep: if True, data preparation is skipped.

    """

    if skip_prep:
        return

    # If the data folders do not exist, we need to extract the data
    if not os.path.isdir(os.path.join(data_folder, "stop")):
        # Check for zip file and download if it doesn't exist
        zip_location = os.path.join(data_folder, "stop.tar.gz")
        if not os.path.exists(zip_location):
            url = "https://dl.fbaipublicfiles.com/stop/stop.tar.gz"
            download_file(url, zip_location) 
            shutil.unpack_archive(zip_location, data_folder)  # download_file doesn't handle unpacking tar.gz archives properly
        else:
            logger.info("Extracting stop.tar.gz...")
            shutil.unpack_archive(zip_location, data_folder)

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
