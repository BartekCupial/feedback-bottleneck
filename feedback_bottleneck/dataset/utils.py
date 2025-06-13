import logging
import pickle
from pathlib import Path


def load_or_compute(filename, compute_fn):
    if Path(filename).exists():
        with open(filename, "rb") as f:
            logging.info(f"Loading cached data from {filename}")
            return pickle.load(f)

    logging.info(f"Computing data and saving to {filename}")
    result = compute_fn()
    with open(filename, "wb") as f:
        pickle.dump(result, f)
        logging.info(f"Data saved to {filename}")

    return result
