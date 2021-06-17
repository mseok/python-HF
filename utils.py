import logging
from logging.handlers import RotatingFileHandler

import numpy as np
import torch


def read_xyz(fn):
    """\
    Read the .xyz file.
    XYZ format is chemical structure data format.
    For more information, see :any:https://en.wikipedia.org/wiki/XYZ_file_format

    Parameters:
        fn:                 str
            The xyz filename.

    Returns:
        n_atoms:            int
            The number of the atoms in molecule.
        atom_symbols:         list[str]
            Atom symbols of molecule.
        atom_coordinates:   list[list[float]]
            Atom coordinates of molecule.
    """
    with open(fn, "r") as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        n_atom_line = lines[0]
        data_line = lines[2:]
        try:
            n_atoms = int(n_atom_line[0])
        except IndexError or ValueError:
            raise IOError("Invalid file type. Check the file again.")

        atom_symbols = [line[0] for line in data_line]
        atom_coordinates = [list(map(float, line[1:])) for line in data_line]
    return n_atoms, atom_symbols, atom_coordinates


def convert_to_tensor(x):
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return torch.Tensor(x)
    elif isinstance(x, float) or isinstance(x, int):
        return torch.tensor(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        raise TypeError("Invalid type for tensor conversion.")


def initialize_logger(log_file=None, log_file_level=logging.NOTSET, rotate=False):
    log_format = logging.Formatter("%(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != "":
        if rotate:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=1000000, backupCount=10)
        else:
            file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def test():
    n_atoms, atom_symbols, atom_coordinates = read_xyz("./benzene.xyz")
    print(n_atoms)
    print(atom_symbols)
    print(atom_coordinates)


if __name__ == "__main__":
    test()
