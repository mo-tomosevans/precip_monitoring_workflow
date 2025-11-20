import os
import pathlib
import re
import inspect
import itertools
from typing import List, Union
from argparse import ArgumentTypeError
import datetime
import numpy as np
import iris
from iris.exceptions import CoordinateNotFoundError
from oemplotlib import LOGGER as _OEMLOGGER
import oemplotlib
from oemplotlib.loaders import PPLoader

LOGGER = _OEMLOGGER.getChild(__name__)


def check_stash_regex(arg_value):
    stash_pattern = re.compile(r"^m[0-9]{2}s[0-9]{2}i[0-9]{3}$")
    if stash_pattern.match(arg_value):
        return True
    return False


def replace_prob_strings(full_name: str) -> str:
    """Replaces equality operators with alphabetical equivalents.

    Replaces the <=, >=, <, >, and = operators with le, ge,
    lt, gt, eq character equivalents.

    .. warn::
       This uses simple calls to replace. The above mentioned
       characters will be replaced even if they are not actually
       equality operators.

    Args:
        full_name (str): The string in which operators should be replaced.

    Returns:
        str: The modified string.
    """
    return (
        full_name.replace("<=", "le")
        .replace(">=", "ge")
        .replace("<", "lt")
        .replace(">", "gt")
        .replace("=", "eq")
    )


def filesafe_string(
    instring: Union[str, pathlib.Path], extra_characters: List[str] = None
) -> str:
    """Makes a string safe to use as a file name.

    Makes instring safe for use as a file name by replacing . with p, removing whitespace
    and non alphanumeric characters except for _ and -.

    warning:: All . in the instring will be replaced. Instring should not include
              the file name suffix.

    Args:
        instring (Union[str, pathlib.Path]): The string to be made safe.
        extra_characters (List[str], optional): A list of additional characters
                                                that should be kept in the final string.
                                                Defaults to None.

    Returns:
        str: A string that is safe for using as a file name.

    Raises:
        ValueError: Raised if instring includes directories
    """

    extra_characters = [] if extra_characters is None else extra_characters

    testpath = pathlib.Path(instring)
    if os.path.sep not in extra_characters and len(testpath.parts) > 1:
        raise ValueError(
            "String corresponds to file name with directories, "
            "rendering file safe could change directory structure"
        )

    instring = replace_prob_strings(str(instring))

    safe_chars = ["_", "-"]
    safe_chars.extend(extra_characters)

    return "".join(
        [
            f
            for f in "_".join(instring.split()).replace(".", "p")
            if f.isalnum() or f in safe_chars
        ]
    )


def argparse_stash_type(stash: str) -> str:
    """A function to validate stash input codes.

    This function is intended to be used as the type argument of an 
    ArgumentParser.add_argument call. If the input stash matches
    the expected format it will be returned, otherwise an exception will
    be raised.

    Args:
        stash (str): A stash code to be checked

    Raises:
        ArgumentTypeError: Raised if the input stash code is not in the correct format

    Returns:
        str: The input stash code
    """

    if not check_stash_regex(stash):
        raise ArgumentTypeError("stash code not in required mNNsNNiNNN format")
    return stash


def get_lat_lon_from_cube(cube):
    """returns the latitude/longitude or equivalent coordinates

    Searches the cube for latitude/longitude, grid_latitude/grid_longitude
    or projection_y_coordinate/projection_x_coordinate coordinates.
    

    Args:
        cube (iris.cube.Cube): Cube from which coordinates should be extracted

    Raises:
        iris.CoordinateNotFoundError: raised if the cube does not have any
                                      of the expected coordinate pairs

    Returns:
        Tuple[iris.coords.Coord, iris.coords.Coord]: Latitude and longitude coordinates
    """

    coord_name_pairs = [
        ("latitude", "longitude"),
        ("grid_latitude", "grid_longitude"),
        ("projection_y_coordinate", "projection_x_coordinate"),
    ]

    for yname, xname in coord_name_pairs:
        try:
            lat = cube.coord(yname)
            lon = cube.coord(xname)
            return lat, lon
        except CoordinateNotFoundError as exp:
            lasterr = exp
    raise lasterr


def fix_longitude_bounds(lon: float, max_lon: int = 180) -> float:
    """Coerces longitudes in to a specified range

    Args:
        lon (float): The Longitude to be fixed
        max_lon (Literal[180, 360], optional): If 180 the longitude will me moved
                                               in to the range -180 <= longitude <= 360.
                                               If 360 the longitude will me moved
                                               in to the range 0 <= longitude < 360.
                                               Defaults to 180.

    Raises:
        ValueError: If after adjustments the longitude is still not in the correct range.
        ValueError: If max_range is not set correctly.

    Returns:
        float: The corrected longitude.
    """

    if max_lon == 180:
        if lon > 180:
            lon = lon - 360
        elif lon <= -180:
            lon = lon + 360
        if not (-180 < lon <= 180):
            raise ValueError(
                "Unable to coerce longitude to range -180 < longitude <= 180"
            )
        return lon
    if max_lon == 360:
        if lon < 0:
            return lon + 360
        if lon >= 360:
            return lon - 360
        if not (0 <= lon < 360):
            raise ValueError("Unable to coerce longitude to range 0 <= longitude < 360")
        return lon
    raise ValueError("max_lon is not 180 or 360")


def projection_from_cube(cube):
    "returns a cartopy.crs.Projection based from a cube"

    coord, _ = get_lat_lon_from_cube(cube)
    return coord.coord_system.as_cartopy_projection()


def get_filename_template(cube, ens_fname=None, det_fname=None):
    """Returns an appropriate file name template

    Checks if the cube represents and ensemble or not and returns
    the corresponding file name template from the config.

    Args:
        cube (iris.cube.Cube): The cube to be checked
        ens_fname (str, optional): An override for the ensemble file name template.
                                   If specified this will be returned instead
                                   of the value in the config. Defaults to None.
        det_fname (str, optional): An override for the deterministic file name template.
                                   If specified this will be returned instead
                                   of the value in the config. Defaults to None.

    Returns:
        str: An appropriate file name template for the given cube.
    """

    try:
        cube.slices_over("realization")
        fname = ens_fname if ens_fname else oemplotlib.CONFIG["strings"]["ENS_FNAME"]
    except iris.exceptions.CoordinateNotFoundError:
        fname = det_fname if det_fname else oemplotlib.CONFIG["strings"]["DET_FNAME"]
    return fname


def ens_safe_slicer(cube, ens_fname=None, det_fname=None):
    try:
        rslices = cube.slices_over("realization")
    except iris.exceptions.CoordinateNotFoundError:
        rslices = [cube]
    fname = get_filename_template(cube, ens_fname=ens_fname, det_fname=det_fname)
    return zip(rslices, itertools.repeat(fname))


def expand_range_string(instr):
    if ".." not in instr or not (
        instr.startswith("linspace:") or instr.startswith("range:")
    ):
        raise ValueError(f"input string {instr} not in correct format for expansion")
    if instr.startswith("linspace:"):
        instr = instr[9:].strip()
        expand = np.linspace
        convert = float
    if instr.startswith("range:"):
        instr = instr[6:].strip()
        expand = range
        convert = int
    LOGGER.debug("Expanding str %s with %s", instr, expand)
    vals = instr.split("..")
    if len(vals) not in [2, 3]:
        raise ValueError("input string expanded to wrong number of variables")
    vals[0] = convert(vals[0])
    vals[1] = convert(vals[1])
    if len(vals) == 3:
        vals[2] = int(vals[2])
    return expand(*vals)


def expand_range_string_list(strlist):
    if not isinstance(strlist, list):
        raise ValueError("input must be a list")
    LOGGER.debug("Expanding str linspace list: %s", strlist)
    outlist = []
    for s in strlist:
        if isinstance(s, str):
            outlist.extend(expand_range_string(s))
        else:
            outlist.append(s)
    return outlist


def time_window_generator(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    interval: datetime.timedelta,
    lower_bound_offset: datetime.timedelta = None,
    upper_bound_offset: datetime.timedelta = None,
    cumulative: bool = False,
):
    """Generate a sequence of time points and (optionally) bounds.

    Args:
        start_time (datetime.datetime): The start time of the sequence
        end_time (datetime.datetime): The end time of the sequence.
                                      Note: this is not guaranteed to be in the
                                      final sequence returned.
        interval (datetime.timedelta): The interval between entries in the sequence.
        lower_bound_offset (datetime.timedelta, optional): The offset between the point in the sequence
                                                           and it's lower bound
                                                           Note that for a lower bound earlier than the point
                                                           this should be a positive timedelta.
                                                           Defaults to None.
        upper_bound_offset (datetime.timedelta, optional): The offset between the point in the sequence
                                                           and it's upper bound.
                                                           Defaults to None.
        cumulative (bool, optional): If set to True the lower_bound will be the same
                                     for all entries of the sequence returned.
                                     Defaults to False.

    Raises:
        ValueError: Raised if only one of lower_bound_offset or upper_bound_offset is set.

    Yields:
        [datetime.datetime, tuple]: If lower/upper_bound_offset are set this will be a tuple of
                                    (point, lower_bound, upper bound) each defined as a
                                    datetime.datetime. Otherwise only the point will be
                                    returned as a datetime.datetime.
    """

    bounds = [lower_bound_offset is not None, upper_bound_offset is not None]
    if any(bounds) and not all(bounds):
        raise ValueError(
            "Either both lower and upper bounds offsets must be specified or neither"
        )

    dtime = start_time
    if lower_bound_offset:
        # both bounds should be provided
        lower_bound = start_time - lower_bound_offset
        upper_bound = start_time + upper_bound_offset
        while dtime <= end_time:
            yield dtime, lower_bound, upper_bound
            dtime += interval
            upper_bound += interval
            if not cumulative:
                lower_bound += interval
    else:
        while dtime <= end_time:
            yield dtime
        dtime += interval


def get_all_plot_handlers(
    indict: dict,
    allowlist: List[str] = None,
    denylist: List[str] = None,
    required_type=oemplotlib.plots.PlotHandlerABC,
) -> List:
    """Find all plotting Classes and return them in a list

    Args:
        indict (dict): Dict of {name: object} pairs. i.e. as generated by globals()
        allowlist (List[str], optional): If provided, only objects whose name is in
                                         the allowlist will be returned.
        denylist (List[str], optional): If provided, objects whose name is in
                                        the denylist will be returned. The name
                                        may be a regular expression.
        required_type (class, optional): The type of plot handler to be plotted.

    Returns:
        List: List if class instances that can be used to make plots.
    """
    LOGGER.debug("Extracting plot handlers from %s", indict)
    plotters = set()

    denylist = [] if denylist is None else denylist

    for name, item in indict.items():
        do_plot = True
        if allowlist and not any([re.match(allow, name) for allow in allowlist]):
            do_plot = False
        for deny in denylist:
            if re.match(deny, name):
                LOGGER.warning("SKIPPING %s, matches denylist entry %s", name, deny)
                do_plot = False
                break
        if name.startswith("_"):
            do_plot = False
        if do_plot and inspect.isclass(item) and issubclass(item, required_type):
            plotters.add(indict[name])
    return [p() for p in plotters]


def get_plot_stash(plotter):
    if isinstance(plotter, oemplotlib.plots.PlotHandlerABC):
        return list(set(plotter.required_stash))
    else:
        stash = []
        for p in plotter:
            stash += p.required_stash
        return list(set(stash))


def plot_all_PP_plot_handlers(
    handlers_dict,
    input_dir,
    input_files,
    output_dir,
    stash_allowlist=None,
    imt_extra_tags=None,
    imt_db_dir=None,
    loader=PPLoader,
    grids=None,
    model=None,
    model_config_dir=None,
    allowlist=None,
    denylist=None,
    fail_on_exception=False,
    extra_plot_handler_kwargs=None,
):

    if extra_plot_handler_kwargs is None:
        extra_plot_handler_kwargs = {}

    plotters = oemplotlib.utils.get_all_plot_handlers(
        handlers_dict,
        allowlist=allowlist,
        denylist=denylist,
        required_type=oemplotlib.plots.PPPlotHandlerABC,
    )
    LOGGER.debug("found plotters: %s", plotters)
    stash = oemplotlib.utils.get_plot_stash(plotters)
    LOGGER.debug("All plots would require stash %s", stash)

    if stash_allowlist:
        stash = [s for s in stash if s in stash_allowlist]

    if len(stash) < 1:
        raise ValueError("No stash requested, no plots will be made")

    LOGGER.info("Making plots using stash %s", stash)

    all_input_files = []
    if not isinstance(input_dir, list):
        input_dir = [input_dir]
    for indir in input_dir:
        for f in input_files:
            if len(list(indir.glob(str(f)))) > 0:
                all_input_files.append(indir / f)
    if not all_input_files:
        raise IOError("No input files found")

    cubes = loader(
        all_input_files, iris.AttributeConstraint(STASH=lambda s: s in stash),
    ).load()

    if len(cubes) < 1:
        raise ValueError("No cubes loaded, no plots will be made")

    LOGGER.info("Loaded cubes:\n%s", cubes)

    plot_tags = imt_extra_tags.copy() if imt_extra_tags else {}
    if model is None:
        model = plot_tags.pop("model", None)
    plot_tags["model"] = model

    def plotters_loop(cubes, plot_extra_tags, regridder=None):
        plotting_succeeded = False
        num_plotters = len(plotters)
        for i, p in enumerate(plotters):
            LOGGER.info(
                "Starting plotting with plotter %s of %s: %s",
                i + 1,
                num_plotters,
                p.__class__.__name__,
            )
            try:
                p.plot(
                    cubes,
                    output_dir,
                    plot_extra_tags,
                    imt_db_dir,
                    regridder=regridder,
                    **extra_plot_handler_kwargs,
                )
                plotting_succeeded = True
            except Exception as ex:
                LOGGER.error(
                    "Unable to plot %s, caught exception: %s", p.__class__.__name__, ex
                )
                if fail_on_exception:
                    raise ex
        return plotting_succeeded

    any_plotting_succeeded = False
    if grids:
        if not model_config_dir:
            raise ValueError(
                "regridding requested but model config directory not provided"
            )
        for grid in grids:
            LOGGER.info("Plotting on grid %s", grid)
            regridder = oemplotlib.gridtools.SimpleRegridder(
                cubes[0],
                oemplotlib.gridtools.GridfileManager().load_grid(
                    model_config_dir, grid
                ),
                target_grid_name=grid,
            )
            plot_tags["grid"] = grid
            any_plotting_succeeded = (
                plotters_loop(cubes, plot_tags, regridder=regridder)
                or any_plotting_succeeded
            )
    else:
        LOGGER.info("Model grids not specified, defaulting to native grid")
        if model:
            plot_tags["grid"] = model
        else:
            plot_tags["grid"] = "???"
        regridder = oemplotlib.gridtools.NullRegridder(cubes[0], plot_tags["grid"])
        any_plotting_succeeded = plotters_loop(cubes, plot_tags, regridder=regridder)
    if not any_plotting_succeeded:
        raise RuntimeError("All plotting classes failed to make plots")


def plot_all_MetDB_plot_handlers(
    handlers_dict,
    input_dir,
    input_files,
    output_dir,
    imt_extra_tags=None,
    imt_db_dir=None,
    grids=None,
    model=None,
    model_config_dir=None,
    allowlist=None,
    denylist=None,
    fail_on_exception=False,
    extra_plot_handler_kwargs=None,
    regridders_setup_callback=None,
):

    if extra_plot_handler_kwargs is None:
        extra_plot_handler_kwargs = {}
    else:
        extra_plot_handler_kwargs = extra_plot_handler_kwargs.copy()

    plotters = oemplotlib.utils.get_all_plot_handlers(
        handlers_dict,
        allowlist=allowlist,
        denylist=denylist,
        required_type=oemplotlib.plots.MetDBPlotHandlerABC,
    )
    LOGGER.debug("found plotters: %s", plotters)

    cubes = iris.load([str(input_dir / f) for f in input_files],)

    if len(cubes) < 1:
        raise ValueError("No cubes loaded, no plots will be made")

    LOGGER.info("Loaded cubes:\n%s", cubes)

    plot_tags = imt_extra_tags.copy() if imt_extra_tags else {}
    if model is None:
        model = plot_tags.pop("model", None)
    plot_tags["model"] = None
    plot_tags["member"] = None

    def plotters_loop(cubes, plot_extra_tags, grid=None):
        plotting_succeeded = False
        num_plotters = len(plotters)
        for i, p in enumerate(plotters):
            LOGGER.info(
                "Starting plotting with plotter %s of %s: %s",
                i + 1,
                num_plotters,
                p.__class__.__name__,
            )
            try:
                p.plot(
                    cubes,
                    output_dir,
                    plot_extra_tags,
                    imt_db_dir,
                    grid=grid,
                    **extra_plot_handler_kwargs,
                )
                plotting_succeeded = True
            except Exception as ex:
                LOGGER.error(
                    "Unable to plot %s, caught exception: %s", p.__class__.__name__, ex
                )
                if fail_on_exception:
                    raise ex
        return plotting_succeeded

    any_plotting_succeeded = False
    if grids:
        if not model_config_dir:
            raise ValueError("grids specified but model config directory not provided")
        for grid in grids:
            LOGGER.info("Plotting on grid %s", grid)
            grid_cube = oemplotlib.gridtools.GridfileManager().load_grid(
                model_config_dir, grid
            )
            if regridders_setup_callback is not None:
                extra_plot_handler_kwargs["regridders"] = regridders_setup_callback(
                    cubes[0], grid_cube, grid,
                )
            plot_tags["grid"] = grid
            any_plotting_succeeded = (
                plotters_loop(cubes, plot_tags, grid=grid_cube)
                or any_plotting_succeeded
            )
    else:
        LOGGER.info("Model grids not specified, defaulting to native grid")
        if model:
            plot_tags["grid"] = model
        else:
            plot_tags["grid"] = "???"
        any_plotting_succeeded = plotters_loop(cubes, plot_tags)
    if not any_plotting_succeeded:
        raise RuntimeError("All plotting classes failed to make plots")
