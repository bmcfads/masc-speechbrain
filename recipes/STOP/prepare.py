import logging
import os
import shutil

from speechbrain.dataio.dataio import read_audio
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

    manifest_path = "stop/manifests"
    audio_path = "stop"

    splits = [
        "train",
        "eval",
        "test"
    ]
    ID_start = 0
    for split in splits:
        new_filename = os.path.join(save_folder, split) + f"-all-type={type}.csv"
        if os.path.exists(new_filename):
            continue
        logger.info(f"Preparing {new_filename}...")

        ID = []
        duration = []

        wav = []
        wav_format = []
        wav_opts = []

        semantics = []
        semantics_format = []
        semantics_opts = []

        transcript = []
        transcript_format = []
        transcript_opts = []

        filename = os.path.join(data_folder, manifest_path, split) + ".tsv"
        df = pd.read_csv(filename, sep="\t")
        for i in range(len(df)):
            ID.append(ID_start + i)
            audio_filename = os.path.join(data_folder, audio_path, df.file_id[i].replace(f"_{split}_0", f"_{split}"))
            signal = read_audio(audio_filename)
            duration.append(signal.shape[0] / 16000)

            wav.append(audio_filename)
            wav_format.append("wav")
            wav_opts.append(None)

            semantics.append(df.decoupled_normalized_seqlogical[i])
            semantics_format.append("string")
            semantics_opts.append(None)

            transcript.append(df.normalized_utterance[i])
            transcript_format.append("string")
            transcript_opts.append(None)

        new_df = pd.DataFrame(
            {
                "ID": ID,
                "duration": duration,
                "wav": wav,
                "semantics": semantics,
                "transcript": transcript,
            }
        )

        new_df.to_csv(new_filename, index=False)
        ID_start += len(df)
