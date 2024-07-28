import sys

from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main

if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    show_results_every = 100  # Plots results every N iterations

    # Create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing STOP)
    from prepare import prepare_STOP  # noqa

    # Multi-GPU (DDP) save data preparation
    run_on_main(
        prepare_STOP,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "type": "direct",
            "train_domains": hparams["train_domains"],
            "flat_intents": hparams["flat_intents"],
            "skip_prep": hparams["skip_prep"],
        },
    )
