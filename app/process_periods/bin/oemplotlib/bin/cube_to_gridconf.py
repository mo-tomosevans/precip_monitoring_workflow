#!/usr/bin/env python3

import sys
from typing import IO
import argparse
import iris

from oemplotlib.gridtools import GridfileManager

OPTIONAL_STASH_LIST = [
    "m01s03i236",
    "m01s00i409",
    "m01s04i203",
    "m01s04i204",
    "m01s05i205",
    "m01s05i206",
    "m01s16i222",
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments

    Returns:
        argparse.Namespace: A namespace with the parsed argument values.
    """

    import pathlib
    from oemplotlib.utils import argparse_stash_type

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stash_codes",
        type=argparse_stash_type,
        nargs="*",
        required=False,
        default=OPTIONAL_STASH_LIST,
        help=(
            "List of stash that will be loaded to try and extract a grid from "
            "the given file(s). Note that if no grid is found using these stash codes "
            "the code will retry loading all cubes in the file(s)."
        ),
    )
    parser.add_argument(
        "--input_files",
        type=pathlib.Path,
        nargs="*",
        required=True,
        help="List of files to be searched for cubes.",
    )

    parser.add_argument(
        "--reverse",
        action="store_true",
        help=(
            "Reverse normal operation, read a grid config file and print a cube. "
            "This is intended for validating grid files."
        ),
    )

    return parser.parse_args()


def output_grid_file(cube: iris.cube.Cube, outstream: IO[str]):
    """Writes grid information from a cube to outstream in a config (ini) format.

    Given an iris cube, this extracts the horizontal grid information, converts it to
    a configobj ini style format and writes it to outstream.

    Args:
        cube (iris.cube.Cube): A cube with a horizontal grid.
        outstream (IO[str]): An object with a write method that takes a
                             string as its only argument. e.g. sys.stout
    """

    grid_manager = GridfileManager()
    stash = cube.attributes.get("STASH", "STASH NOT DEFINED")

    outstream.write(
        f"# grid config produced using stash {stash} using cube_to_gridconf.py\n"
    )
    outstream.write(grid_manager.conf_from_cube(cube))
    outstream.write("\n")


def cube_from_gridfile(gridfile: str, outstream: IO[str]):
    """Processes gridfile and writes the cube produced and its coords to outstream

    Args:
        gridfile (str): Path to the grid specification file to be read
        outstream (IO[str]): An object with a write method that takes a
                             string as its only argument. e.g. sys.stout
    """
    grid_manager = GridfileManager()
    cube = grid_manager.cube_from_conf(gridfile)
    outstream.write(str(cube))
    outstream.write("\n")
    for coord in cube.coords():
        outstream.write(str(coord))
        outstream.write("\n")


def main():
    """Reads an iris cube from a given file(s) and writes it's grid definition to a file.

    Given a list of files, this will take the first cube it can find and use that to write
    an oemplotlib grid definition file. To speed up loading, each file will be checked in
    turn for a limited set of STASH codes. If no grid is found, each will be loaded in turn
    without any constraints until a cube is found.
    """

    # first ensure nothing is output to stdout by mistake
    stdout = sys.stdout
    sys.stdout = sys.stderr
    import logging
    import iris

    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    args = parse_args()
    input_files = []
    for f in args.input_files:
        fdir = f.parent
        ffile = f.name
        input_files += fdir.resolve().glob(ffile)
    input_files = [str(s) for s in input_files]
    logger.debug("Found input files:\n%s", input_files)

    if args.reverse:
        for f in input_files:
            cube_from_gridfile(f, stdout)
    else:
        # first pass, use constrained loading
        for f in input_files:
            cubes = iris.load(
                f, iris.AttributeConstraint(STASH=lambda s: s in args.stash_codes)
            )
            if cubes:
                output_grid_file(cubes[0], stdout)
                break
        else:
            logger.info("No cubes found, trying again using unconstrained loading")
            # second pass, load all cubes in the file
            for f in input_files:
                cubes = iris.load(f)
                if cubes:
                    output_grid_file(cubes[0], stdout)
                    break
            else:
                # no cubes found
                raise ValueError("Unable to find cubes in files")


if __name__ == "__main__":
    main()
