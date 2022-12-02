import dotenv
import hydra
from omegaconf import DictConfig
import pyrootutils

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

# project root setup
# searches for root indicators in parent dirs, like ".git", "pyproject.toml", etc.
# sets PROJECT_ROOT environment variable (used in `configs/paths/default.yaml`)
# loads environment variables from ".env" if exists
# adds root dir to the PYTHONPATH (so this file can be run from any place)
# https://github.com/ashleve/pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

@hydra.main(config_path="configs/", config_name="main.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.experiment.classification.trn_val_tst.sa import process

    # Applies optional utilities
    utils.extras(config)

    # Train model
    return process(config)


if __name__ == "__main__":
    main()
