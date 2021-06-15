def read_xyz(fn):
    """\
    Read the .xyz file.
    XYZ format is chemical structure data format.
    For more information, see :any:https://en.wikipedia.org/wiki/XYZ_file_format

    Parameters:
    fn: xyz filename

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


def test():
    n_atoms, atom_symbols, atom_coordinates = read_xyz("./benzene.xyz")
    print(n_atoms)
    print(atom_symbols)
    print(atom_coordinates)


if __name__ == "__main__":
    test()
