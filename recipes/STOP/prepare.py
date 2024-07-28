import logging
import os
import shutil

from speechbrain.dataio.dataio import merge_csvs, read_audio
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


def prepare_STOP(data_folder, save_folder, type, train_domains=[], flat_intents=False, skip_prep=False):
    """
    This function prepares the STOP dataset.
    If the folder does not exist, the zip file will be extracted. If the zip file does not exist, it will be downloaded.

    data_folder : path to STOP dataset.
    save_folder: path where to save the csv manifest files.
    slu_type : one of the following:

      "direct":{input=audio, output=semantics}

    train_domains : list of domain to include; if empty, all domains included.
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

    manifest_dir = os.path.join(data_folder, "stop/manifests")
    sb_manifest_dir = os.path.join(data_folder, manifest_dir, "speechbrain")
    audio_dir = os.path.join(data_folder, "stop")

    if not os.path.isdir(sb_manifest_dir):
        os.makedirs(sb_manifest_dir)

    ID_start = 0

    domains = [
        "alarm",
        "event",
        "messaging",
        "music",
        "navigation",
        "reminder",
        "timer",
        "weather",
    ]

    splits = [
        "train",
        "eval",
        "test",
    ]

    # Prepare all domains manifest files.
    for split in splits:
        new_filename = os.path.join(sb_manifest_dir, split) + f"---type={type}.csv"
        if os.path.exists(new_filename):
            continue
        logger.info(f"Preparing {new_filename}...")

        ID = []
        duration = []

        wav = []
        wav_format = []
        wav_opts = []

        domain = []

        semantics = []
        semantics_format = []
        semantics_opts = []

        transcript = []
        transcript_format = []
        transcript_opts = []

        filename = os.path.join(manifest_dir, split) + ".tsv"
        df = pd.read_csv(filename, sep="\t")
        for i in range(len(df)):
            ID.append(ID_start + i)
            audio_filename = os.path.join(audio_dir, df.file_id[i].replace(f"_{split}_0", f"_{split}"))
            signal = read_audio(audio_filename)
            duration.append(signal.shape[0] / 16000)

            wav.append(audio_filename)
            wav_format.append("wav")
            wav_opts.append(None)

            domain.append(df.domain[i])

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
                "domain": domain,
                "semantics": semantics,
                "transcript": transcript,
            }
        )

        new_df.to_csv(new_filename, index=False)
        ID_start += len(df)

    # Prepare domain and flat intents specific manifest files.
    for split in splits:
        base_filename = os.path.join(sb_manifest_dir, split)
        df_all = pd.read_csv(base_filename + f"---type={type}.csv")
        df_flat = df_all[df_all["semantics"].str.count("IN:") == 1]
        df_flat.to_csv(base_filename + f"---flat-type={type}.csv", index=False)

        for domain in domains:
            domain_filename = base_filename + f"-{domain}-type={type}.csv"
            flat_filename = base_filename + f"-{domain}-flat-type={type}.csv"

            if not os.path.exists(domain_filename):
                logger.info(f"Preparing {domain_filename}...")
                df_domain = df_all[df_all["domain"] == domain]
                df_domain.to_csv(domain_filename, index=False)

            if not os.path.exists(flat_filename):
                logger.info(f"Preparing {flat_filename}...")
                df_domain_flat = df_flat[df_flat["domain"] == domain]
                df_domain_flat.to_csv(flat_filename, index=False)

    # Merge and save the .csv files for model training / testing
    if flat_intents:
        if train_domains:
            train_csv_files = [f"train-{domain}-flat-type={type}.csv" for domain in train_domains]
            eval_csv_files = [f"eval-{domain}-flat-type={type}.csv" for domain in train_domains]
            test_csv_files = [f"test-{domain}-flat-type={type}.csv" for domain in train_domains]
        else:
            train_csv_files = [f"train---flat-type={type}.csv"]
            eval_csv_files = [f"train---flat-type={type}.csv"]
            test_csv_files = [f"train---flat-type={type}.csv"]
    else:
        if train_domains:
            train_csv_files = [f"train-{domain}-type={type}.csv" for domain in train_domains]
            eval_csv_files = [f"eval-{domain}-type={type}.csv" for domain in train_domains]
            test_csv_files = [f"test-{domain}-type={type}.csv" for domain in train_domains]
        else:
            train_csv_files = [f"train---type={type}.csv"]
            eval_csv_files = [f"train---type={type}.csv"]
            test_csv_files = [f"train---type={type}.csv"]

    train_filename = f"train-type={type}.csv"
    eval_filename = f"eval-type={type}.csv"
    test_filename = f"test-type={type}.csv"

    merge_csvs(sb_manifest_dir, train_csv_files, train_filename)
    merge_csvs(sb_manifest_dir, eval_csv_files, eval_filename)
    merge_csvs(sb_manifest_dir, test_csv_files, test_filename)

    # merge_csvs() only saves to the source directory.
    # Copy merged csvs from data folder to experiments folder.
    shutil.copyfile(os.path.join(sb_manifest_dir, train_filename), os.path.join(save_folder, train_filename))
    shutil.copyfile(os.path.join(sb_manifest_dir, eval_filename), os.path.join(save_folder, eval_filename))
    shutil.copyfile(os.path.join(sb_manifest_dir, test_filename), os.path.join(save_folder, test_filename))
    os.remove(os.path.join(sb_manifest_dir, train_filename))
    os.remove(os.path.join(sb_manifest_dir, eval_filename))
    os.remove(os.path.join(sb_manifest_dir, test_filename))
