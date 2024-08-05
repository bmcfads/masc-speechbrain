"""Recipe for training a unigram tokenizer with STOP.
The tokenizer converts semantics into sub-word units that can
be used to train a language (LM) or an acoustic model (AM).

To run this recipe, do the following:
> python train.py hparams/train.yaml


Authors
 * Brian McFadden 2024
 (original: Timers-and-Such
   * Abdel Heba 2021
   * Mirco Ravanelli 2021
   * Loren Lugosch 2021)
"""

import sys

from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main

if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create ddp_group with the right communication protocol.
    sb.utils.distributed.ddp_init_group(run_opts)

    # Dataset prep (parsing STOP).
    from prepare import prepare_STOP  # noqa

    # Create experiment directory.
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Multi-gpu (DDP) save data preparation.
    run_on_main(
        prepare_STOP,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "type": "direct",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Train tokenizer.
    hparams["tokenizer"]()
