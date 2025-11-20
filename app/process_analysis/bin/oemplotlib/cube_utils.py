import datetime
from typing import Union, Callable, List, Generator
import functools
import iris
import iris.analysis
import numpy as np

from oemplotlib import (
    DEFAULT_CUBE_MEMBER_COORD,
    DEFAULT_CUBE_TIME_COORD,
    DEFAULT_CUBE_FC_PERIOD_COORD,
    DEFAULT_CUBE_FC_REF_COORD,
    LOGGER,
)
import oemplotlib
from oemplotlib.utils import time_window_generator

SECOND_AS_HOUR_FRACTION = 1.0 / 3600.0
SECOND_AS_MINUTE_FRACTION = 1.0 / 60.0

LOGGER = LOGGER.getChild(__name__)


def cube_or_cubelist(cubefn) -> Callable:
    @functools.wraps(cubefn)
    def inner(
        c_or_clist: Union[iris.cube.Cube, iris.cube.CubeList], *args, **kwargs
    ) -> Union[iris.cube.Cube, iris.cube.CubeList]:
        if isinstance(c_or_clist, iris.cube.Cube):
            return cubefn(c_or_clist, *args, **kwargs)
        elif isinstance(c_or_clist, iris.cube.CubeList):
            results = [cubefn(cube, *args, **kwargs) for cube in c_or_clist]
            return iris.cube.CubeList([r for r in results if r is not None])
        else:
            raise ValueError("First argument muse be a Cube or a CubeList")

    return inner


def is_instantaneous(cube: iris.cube.Cube):
    """
    Returns True if the input :class:`iris.cube.Cube` object has no cell \
    methods associated with its time dimension.
    """
    for cell_method in cube.cell_methods:
        if "time" in cell_method.coord_names:
            return False
    return True


def is_centred_coord(coord: iris.coords.Coord) -> bool:
    """Tests if an iris coordinate has bounds centred on a point

    Args:
        coord (iris.coord.Coord): The coordinate to be tested

    Returns:
        bool: True if all of the cells of the coord have bounds centred
              around their point otherwise False
    """

    if not coord.has_bounds():
        return False

    cells = coord.cells()

    test = [
        cell.point == cell.bound[0] + (cell.bound[1] - cell.bound[0]) / 2.0
        for cell in cells
    ]
    return all(test)


def is_running_time(
    cube: iris.cube.Cube, time_coord_name: str = DEFAULT_CUBE_TIME_COORD
) -> bool:
    """Returns True if the cube time is for a running time.

    Args:
        cube (iris.cube.Cube): The cube to be checked
        time_coord_name (str, optional): name of the time coordinate to be checked.
                                         Defaults to DEFAULT_CUBE_TIME_COORD.

    Returns:
        (boolean): True if the time coordinate is a running time, else False
    """
    if (
        cube.coord(time_coord_name).has_bounds()
        and len(cube.coord(time_coord_name).points) > 1
        and len(set(cube.coord(time_coord_name).bounds[:, 0])) == 1
    ):
        return True
    return False


def extract_common_member_times(
    incube: Union[iris.cube.Cube, iris.cube.CubeList],
    time_coord_name: str = DEFAULT_CUBE_TIME_COORD,
) -> Union[iris.cube.Cube, iris.cube.CubeList]:
    """Extract the period of time over which all members overlap.

    Args:
        incube (Union[iris.cube.Cube, iris.cube.CubeList]): The cube or CubeList from which
                                                            to extract a time period.
        time_coord_name (str, optional): The name of the time coordinate to be used
                                         in the extraction.
                                         Defaults to oemplotlib.DEFAULT_CUBE_TIME_COORD.

    Raises:
        ValueError: Raised if the cubes do not all share a common overlapping period.

    Returns:
        iris.cube.Cube or iris.cube.CubeList: The subset of the cube or CubeList over which
                                              all members overlap in time. If possible a Cube
                                              will be returned, otherwise a CubeList.
    """

    if not isinstance(incube, iris.cube.CubeList):
        incube = iris.cube.CubeList([incube])

    starts = []
    ends = []
    for cube in incube:
        try:
            cube.coord(oemplotlib.DEFAULT_CUBE_MEMBER_COORD)
        except iris.iris.exceptions.CoordinateNotFoundError as exp:
            # not an ensemble
            raise ValueError(
                "extract_common_member_times: cube has no "
                f"{oemplotlib.DEFAULT_CUBE_MEMBER_COORD} coordinate."
            ) from exp
        members = set(cube.coord(oemplotlib.DEFAULT_CUBE_MEMBER_COORD).points)

        for mem in members:
            tmpdict = {oemplotlib.DEFAULT_CUBE_MEMBER_COORD: mem}
            mem_cube = cube.extract(iris.Constraint(**tmpdict))
            tcoord = mem_cube.coord(time_coord_name)

            start = tcoord.cell(0).point
            end = tcoord.cell(len(tcoord.points) - 1).point

            LOGGER.debug(
                "extract_common_member_times: mem %s start %s end %s", mem, start, end
            )
            starts.append(start)
            ends.append(end)
    start = max(starts)
    end = min(ends)
    LOGGER.debug("extract_common_member_times: overlapping start %s end %s", start, end)
    if start > end:
        raise ValueError("Cube times do not overlap")
    extracted = extract_time_window(incube, start, end, time_coord_name=time_coord_name)
    if isinstance(extracted, iris.cube.CubeList):
        try:
            outcube = extracted.merge_cube()
        except iris.exceptions.MergeError:
            outcube = extracted.merge()
        return outcube
    return extracted


@cube_or_cubelist
def extract_time_window(
    cube: iris.cube.Cube,
    start_time: datetime.datetime = None,
    end_time: datetime.datetime = None,
    time_coord_name: str = DEFAULT_CUBE_TIME_COORD,
) -> iris.cube.Cube:
    """Extracts a time window from a cube

    Args:
        cube (iris.cube.Cube): The cube from which data should be extracted
        start_time (datetime.datetime, optional): The start of the time window
        end_time (datetime.datetime, optional): The end of the time window
        time_coord_name (str, optional): The name of the cube coordinate on which the cube should be constrained.
                                         Defaults to DEFAULT_CUBE_TIME_COORD.

    Returns:
        iris.cube.Cube: A copy of the cube cube constrained to only contain data over the required window.
    """

    if start_time and end_time:
        tmp_kwargs = {
            time_coord_name: lambda cell: start_time <= cell.point <= end_time
        }
    elif start_time:
        tmp_kwargs = {time_coord_name: lambda cell: start_time <= cell.point}
    elif end_time:
        tmp_kwargs = {time_coord_name: lambda cell: cell.point <= end_time}
    else:
        LOGGER.warning(
            "No start or and times specified for time window, "
            "returning copy of input cube"
        )
        return cube.copy()
    extracted = cube.extract(iris.Constraint(**tmp_kwargs))
    if extracted:
        return extracted.copy()
    else:
        return None


@cube_or_cubelist
def thin_lat_lon(cube: iris.cube.Cube, lat_step: int = 1, lon_step: int = 1):
    latdim, londim = oemplotlib.utils.get_lat_lon_from_cube(cube)
    lat_index = cube.coord_dims(latdim)
    if len(lat_index) > 1:
        raise ValueError("Unable to thin multi-dimensional Latitude coordinate")
    lat_index = lat_index[0]
    lon_index = cube.coord_dims(londim)
    if len(lon_index) > 1:
        raise ValueError("Unable to thin multi-dimensional longitude coordinate")
    lon_index = lon_index[0]
    slices = []
    for i, _ in enumerate(cube.dim_coords):
        if i == lat_index:
            slices.append(slice(0, -1, lat_step))
        elif i == lon_index:
            slices.append(slice(0, -1, lon_step))
        else:
            slices.append(slice(None, None))
    return cube[tuple(slices)]


def times_in_order(
    cube: iris.cube.Cube, time_coord_name: str = DEFAULT_CUBE_FC_PERIOD_COORD
) -> bool:
    """Returns true if the time coord is monotonic and in order

    This checks that the time array is increasing and in order,
    but less stringent check than required for an iris.coord.DimCoord
    as it only checks the time 'point's.

    Args:
        cube (iris.cube.Cube): The cube to be checked
        time_coord_name (str, optional): The name of the dim to be checked.
                                         Defaults to DEFAULT_CUBE_FC_PERIOD_COORD.

    Returns:
        bool: True if the time points are in order, else false
    """
    last_time = None
    time_coord = cube.coord(time_coord_name)
    for time in time_coord.cells():
        if last_time and time.point <= last_time:
            return False
        last_time = time.point
    return True


def coord_to_precision(coord: iris.coords.Coord, precision=np.float64) -> None:
    """Converts coord in place to a given precision

    Args:
        coord (iris.coords.Coord): The coord for which the points and bounds
                                   (if present) will be converted.
        precision (numpy precision, optional): The precision to which the coord
                                               should be converted.
                                               Defaults to numpy.float64.
    """
    coord.points = coord.points.astype(precision)
    if coord.has_bounds():
        coord.bounds = coord.bounds.astype(precision)


@cube_or_cubelist
def fix_time_precision(
    cube: iris.cube.Cube,
    precision=np.float64,
    time_coord_name: str = DEFAULT_CUBE_TIME_COORD,
    forecast_period_coord_name: str = DEFAULT_CUBE_FC_PERIOD_COORD,
    forecast_ref_coord_name: str = DEFAULT_CUBE_FC_REF_COORD,
):
    """Set all time coords to the given precision.

    Float precision is required for sub-hourly times so by default
    the precision is set to float64.

    Args:
        cube (iris.cube.Cube): The iris cube to be fixed
        precision (numpy precision, optional): The required precision.
                                               Defaults to numpy.float64.

    Returns: a copy of cube with updated time coords.
    """

    cube = cube.copy()

    try:
        coord_to_precision(cube.coord(forecast_period_coord_name), precision=precision)
    except iris.exceptions.CoordinateNotFoundError:
        pass
    try:
        coord_to_precision(cube.coord(time_coord_name), precision=precision)
    except iris.exceptions.CoordinateNotFoundError:
        pass
    try:
        coord_to_precision(cube.coord(forecast_ref_coord_name), precision=precision)
    except iris.exceptions.CoordinateNotFoundError:
        pass

    return cube


@cube_or_cubelist
def fix_centred_cube_time(
    cube: iris.cube.Cube,
    use_bound: str = "upper",
    coords_to_fix: List[str] = [DEFAULT_CUBE_TIME_COORD, DEFAULT_CUBE_FC_PERIOD_COORD],
    drop_bounds: bool = False,
    time_coord_name: str = DEFAULT_CUBE_TIME_COORD,
    forecast_period_coord_name: str = DEFAULT_CUBE_FC_PERIOD_COORD,
    forecast_ref_coord_name: str = DEFAULT_CUBE_FC_REF_COORD,
) -> iris.cube.Cube:
    """Relabel points in the centre of a period as being valid at the end or start of the period.

    Iris load sometimes labels processing over a period (i.e. max over an hour) as being
    valid in the centre of the period rather than at the one end. This function replaces the
    time 'point' with either it's upper (default) or lower bound.

    .. warning::
       If the time coordinates are not centred a copy of the cube with
       unaltered coordinates will be returned.

    Args:
        cube (iris.cube.Cube): The cube to be fixed
        use_bound (str, "upper" or "lower"): The bound that should be used to replace the centre
                                             point.
        coords_to_fix (List[str], optional): A list of the names of the coordinates that
                                             should be fixed. Defaults to
                                             [DEFAULT_CUBE_TIME_COORD, DEFAULT_CUBE_FC_PERIOD_COORD]
        drop_bounds (bool, optional): If True the returned cube will not have bounds on any coordinates
                                      that have been fixed. Defaults to False.
        time_coord_name (str, optional): The name of the time coordinate.
                                         Defaults to DEFAULT_CUBE_TIME_COORD.
        forecast_period_coord_name (str, optional): 
                                         The name of the forecast period coordinate.
                                         Defaults to DEFAULT_CUBE_FC_PERIOD_COORD.
        forecast_ref_coord_name (str, optional): 
                                         The name of the forecast reference coordinate.
                                         Defaults to DEFAULT_CUBE_FC_REF_COORD.

    Raises:
        ValueError: Raised if use_bound is not "upper" or "lower"
        ValueError: Raised if cube has running times (e.g. running accumulations)
                    or has unbounded coordinates.

    Returns:
        iris.cube.Cube: A copy of the cube with fixed time coordinates.
    """

    if use_bound not in ["upper", "lower"]:
        raise ValueError("use_bound must be 'upper' or 'lower'")
    if use_bound == "upper":
        bound_index = 1
    else:
        bound_index = 0

    # use float precision to handle possible sub-hourly times
    cube = fix_time_precision(
        cube,
        time_coord_name=time_coord_name,
        forecast_period_coord_name=forecast_period_coord_name,
        forecast_ref_coord_name=forecast_ref_coord_name,
    )

    if is_running_time(cube):
        raise ValueError("Unable to handle cubes with running time processing")

    if not all(cube.coord(cname).has_bounds() for cname in coords_to_fix):
        raise ValueError("Unable to fix unbounded coordinates")

    if any(not is_centred_coord(cube.coord(cname)) for cname in coords_to_fix):
        LOGGER.warning(
            "At least one coordinate is not centred, returning unaltered cube"
        )
    else:
        for fix_coord_name in coords_to_fix:
            time_coord = cube.coord(fix_coord_name)
            if drop_bounds:
                new_bounds = None
            else:
                new_bounds = time_coord.bounds
            cube.replace_coord(
                time_coord.copy(
                    points=time_coord.bounds[:, bound_index], bounds=new_bounds
                )
            )

    return cube


@cube_or_cubelist
def fix_running_cube_time(
    cube: iris.cube.Cube,
    time_coord_name: str = DEFAULT_CUBE_TIME_COORD,
    forecast_period_coord_name: str = DEFAULT_CUBE_FC_PERIOD_COORD,
    forecast_ref_coord_name: str = DEFAULT_CUBE_FC_REF_COORD,
) -> iris.cube.Cube:
    """Returns new cube with corrected times for running parameters.

    UMDP F03 does not fully specify how to indicate the data time
    for running time processed fields (e.g. running accumulations)
    that are output at a sub hourly frequency,
    therefore iris incorrectly labels some time coordinates. Even for (whole multiples of)
    hourly accumulations the points on the "time" coordinate can be incorrectly
    positioned half way between the bounds. 

    If the points are half way between the bounds and the metadata describes a running period
    the bounds (which are usually correct) are used to derive the forecast_period
    and forecast_reference_time. If the points are not half way between the
    bounds a number of plausibility tests are carried out on the other time
    metadata coordinates to make sure the metadata looks correct.

    .. note::
       This function assumes the period over which the field has
       been processed ends at the forecast time. e.g. for an average over an
       hour, the hour must be the hour before the given time, not an hour
       centred on the given time.

    Args:
        cube (iris.cube.Cube): An iris cube for a running time processed field

    Raises:
        ValueError: Raised if the time coordinates of the cube do not appear to
                    describe a running period.
        AssertionError: Raised if the cube's metadata does not appear to be broken in
                        the expected way ("time" points are half way between its bounds)
                        but some other time like coordinate is not as expected.
        ValueError: Raised if no suitable forecast reference time can be identified
                    in order to fix the other time coordinates.

    Returns:
        iris.cube.Cube: A copy of the cube with corrected time coordinates
    """

    # sub-hourly requires float precision
    # note, this makes a copy of the cube
    cube = fix_time_precision(
        cube,
        time_coord_name=time_coord_name,
        forecast_period_coord_name=forecast_period_coord_name,
        forecast_ref_coord_name=forecast_period_coord_name,
    )

    if not is_running_time(cube):
        raise ValueError("Cube time dimension is not a running time")

    time_dim = cube.coord_dims(time_coord_name)[0]
    time_coord = cube.coord(time_coord_name)

    # for running accumulations the points usually end up being set as the midpoint of the bounds
    # use this as a safety check
    midpoints = (
        time_coord.bounds[:, 0]
        + (time_coord.bounds[:, 1] - time_coord.bounds[:, 0]) / 2.0
    )
    if ~np.all(np.isclose(time_coord.points, midpoints)):
        # time points not in centre of bound, check if the time coordinates are already correct
        LOGGER.debug(
            "fix_running_cube_time: time coord's points not in the centre of its bounds, "
            "checking metadata"
        )
        err_msg = (
            f"Unable to fix running accumulation: {time_coord_name} coord's points"
            " are not the midpoints of its bounds"
        )
        assert np.all(
            np.isclose(time_coord.points, time_coord.bounds[:, 1])
        ), f"{err_msg} and do not match its upper bounds"
        assert cube.coord(forecast_ref_coord_name).points.shape == (
            1,
        ), f"{err_msg} and {forecast_ref_coord_name} has more than one point"
        assert (
            len(set(cube.coord(forecast_period_coord_name).bounds[:, 0])) == 1
        ), f"{err_msg} and {forecast_period_coord_name} has more than one lower bound"
        assert np.all(
            np.isclose(
                cube.coord(forecast_period_coord_name).points,
                cube.coord(forecast_period_coord_name).bounds[:, 1],
            )
        ), (
            f"{err_msg} and {forecast_period_coord_name} coord's points "
            "do not match its upper bound"
        )
    else:
        LOGGER.debug(
            "fix_running_cube_time: time coord's points in the centre of its bounds, "
            "fixing metadata"
        )

        # replace points with upper bounds
        time_coord = time_coord.copy(
            points=time_coord.bounds[:, 1], bounds=time_coord.bounds
        )
        cube.replace_coord(time_coord)

        # extract and remove original coordinates
        orrig_forecast_period = cube.coord(forecast_period_coord_name)

        cube.remove_coord(orrig_forecast_period)

        # Use forecast reference to recalculate the forecast periods.
        # Assume (require) the forecast period should be equal to the upper bound,
        # e.g. the accumulation up to the validity time.

        # first find an appropriate forecast reference time,
        # validity times on the hour should have the correct reference time
        for cubeslice in cube.slices_over(time_coord_name):
            if cubeslice.coord(time_coord_name).cell(0).bound[1].minute == 0:
                new_forecast_ref_time = cubeslice.coord(forecast_ref_coord_name).copy()
                break
        else:
            raise ValueError(
                "Unable to fix running accumulation: no suitable reference time found"
            )
        cube.remove_coord(cube.coord(forecast_ref_coord_name))
        cube.add_aux_coord(new_forecast_ref_time)

        DT = new_forecast_ref_time.cell(0).point
        forecast_period_points = [
            (c.bound[1] - DT).total_seconds() / 3600.0 for c in time_coord.cells()
        ]

        forecast_period_lower_bound = [
            (c.bound[0] - DT).total_seconds() / 3600.0 for c in time_coord.cells()
        ]
        new_forecast_period = iris.coords.AuxCoord(
            forecast_period_points,
            standard_name=forecast_period_coord_name,
            bounds=np.column_stack(
                (forecast_period_lower_bound, forecast_period_points)
            ),
            units="hours",
        )
        cube.add_aux_coord(new_forecast_period, time_dim)

    return cube


@cube_or_cubelist
def fctime_from_datatime(
    cube: iris.cube.Cube,
    time_coord_name: str = DEFAULT_CUBE_TIME_COORD,
    forecast_period_coord_name: str = DEFAULT_CUBE_FC_PERIOD_COORD,
    forecast_ref_coord_name: str = DEFAULT_CUBE_FC_REF_COORD,
) -> iris.cube.Cube:
    """Calculate a new forecast period dimension.

    Removes the existing forecast period dimension and adds a new one calculated
    from the validity time and the data time.

    .. note::
       This function assumes the the bounds for the validity time (if present) are
       correct and that both the bounds and points for the forecast period can be
       calculated as a difference from the data time.

    Args:
        cube (iris.cube.Cube): The cube from which the new forecast period should be calculated
        time_coord_name (str, optional): Name of the validity time coordinate.
                                         Defaults to DEFAULT_CUBE_TIME_COORD.
        forecast_period_coord_name (str, optional): Name of the forecast period coordinate.
                                                    Defaults to DEFAULT_CUBE_FC_PERIOD_COORD.
        forecast_ref_coord_name (str, optional): Name of the data time coordinate.
                                                 Defaults to DEFAULT_CUBE_FC_REF_COORD.

    Returns:
        iris.cube.Cube: A copy of the cube with a new forecast period coordinate.
    """
    time_dim = cube.coord_dims(time_coord_name)[0]
    time_coord = cube.coord(time_coord_name)
    forecast_reference_times = cube.coord(forecast_ref_coord_name)

    DT = forecast_reference_times.cell(0).point
    forecast_period_points = [
        (c.point - DT).total_seconds() / 3600.0 for c in time_coord.cells()
    ]

    if time_coord.has_bounds():
        forecast_period_upper_bound = [
            (c.bound[1] - DT).total_seconds() / 3600.0 for c in time_coord.cells()
        ]

        forecast_period_lower_bound = [
            (c.bound[0] - DT).total_seconds() / 3600.0 for c in time_coord.cells()
        ]
        new_bounds = np.column_stack(
            (forecast_period_lower_bound, forecast_period_upper_bound)
        )
    else:
        new_bounds = None

    new_forecast_period = iris.coords.AuxCoord(
        forecast_period_points,
        standard_name=forecast_period_coord_name,
        bounds=new_bounds,
        units="hours",
    )
    new_cube = cube.copy()
    try:
        new_cube.remove_coord(forecast_period_coord_name)
    except iris.exceptions.CoordinateNotFoundError:
        pass
    new_cube.add_aux_coord(new_forecast_period, time_dim)
    return new_cube


class _CoordShift:
    def __init__(self, running_parameter=None) -> None:
        self.running_parameter = running_parameter
        self.running_start_time = None

    def _date2num(self, shift_coord, shiftdate):
        if isinstance(shiftdate, datetime.datetime):
            return shift_coord.units.date2num(shiftdate)
        else:
            # not a datetime, just pass through
            return shiftdate

    def shift(self, shift_coord, shift):
        new_points = self._date2num(shift_coord, shift_coord.cell(0).point + shift)
        if shift_coord.has_bounds():
            bounds = shift_coord.cell(0).bound
            if self.running_parameter:
                if self.running_start_time is None:
                    self.running_start_time = self._date2num(
                        shift_coord, bounds[0] + shift
                    )
                lower_bound = self.running_start_time
            else:
                lower_bound = self._date2num(shift_coord, bounds[0] + shift)
            new_bounds = (
                lower_bound,
                self._date2num(shift_coord, bounds[1] + shift),
            )
            new_coord = shift_coord.copy(points=[new_points], bounds=[new_bounds])
        else:
            new_coord = shift_coord.copy(points=[new_points], bounds=None)

        if new_coord.units:
            # Rounding errors can occur after shifting and 0
            # becomes a very small number. Therefore cap
            # the resolution to 1 second.
            thresholds = [
                ("hours", SECOND_AS_HOUR_FRACTION),
                ("minutes", SECOND_AS_MINUTE_FRACTION),
            ]
            for unit_name, threshold in thresholds:
                if new_coord.units == unit_name:
                    below_thresh = np.abs(new_coord.points) < threshold
                    if np.any(below_thresh):
                        LOGGER.debug(
                            "Found sub-threshold (abs(value)<%s)"
                            " points for unit '%s' setting to 0",
                            threshold,
                            unit_name,
                        )
                    new_coord.points = np.where(below_thresh, 0.0, new_coord.points,)
                    if new_coord.has_bounds():
                        below_thresh = np.abs(new_coord.bounds) < threshold
                        if np.any(below_thresh):
                            LOGGER.debug(
                                "Found sub-threshold (abs(value)<%s)"
                                " bounds for unit '%s' setting to 0",
                                threshold,
                                unit_name,
                            )
                        new_coord.bounds = np.where(
                            below_thresh, 0.0, new_coord.bounds,
                        )

        return new_coord


@cube_or_cubelist
def separate_realization_time(
    cube: iris.cube.Cube, forecast_ref_coord_name: str = DEFAULT_CUBE_FC_REF_COORD,
) -> Union[iris.cube.CubeList, iris.cube.Cube]:
    """Separates a multidimensional time+realisation coordinate in to individual coordinates.

    When reading files for multiple members iris will create a multidimensional time
    coordinate that represents the members and the possible time coordinates as a single 
    coordinate. This causes slices_over members or times to slice over each individual 
    combination of member and time, rather than just the specific coordinate requested.
    This function splits the problematic coordinate in to individual dimensions.

    .. note::
       It is not always possible to make a single cube from the split coordinates.
       In this case a CubeList will be returned.

    Args:
        cube (iris.cube.Cube): The cube whose coordinates need to be fixed
        forecast_ref_coord_name (str, optional): The name of the data time coordinate.
                                                 Defaults to DEFAULT_CUBE_FC_REF_COORD.

    Returns:
        Union[iris.cube.CubeList, iris.cube.Cube]: A Cube or CubeList with separated time
                                                   and realization coordinates.
    """
    try:
        cube.coord(oemplotlib.DEFAULT_CUBE_MEMBER_COORD)
    except iris.exceptions.CoordinateNotFoundError:
        pass
    else:
        member_points = list(
            set(cube.coord(oemplotlib.DEFAULT_CUBE_MEMBER_COORD).points)
        )
        member_cubes = [
            cube.extract(iris.Constraint(**{oemplotlib.DEFAULT_CUBE_MEMBER_COORD: mem}))
            for mem in member_points
        ]
        out_cubes = iris.cube.CubeList()
        for c, mem in zip(member_cubes, member_points):
            c.remove_coord(oemplotlib.DEFAULT_CUBE_MEMBER_COORD)
            separated = separate_realization_time(
                c, forecast_ref_coord_name=forecast_ref_coord_name
            )
            if isinstance(separated, iris.cube.CubeList):
                for out in separated:
                    out.add_aux_coord(
                        iris.coords.AuxCoord(
                            np.int32(mem),
                            standard_name=oemplotlib.DEFAULT_CUBE_MEMBER_COORD,
                        )
                    )
                    out_cubes.append(out)
            else:
                separated.add_aux_coord(
                    iris.coords.AuxCoord(
                        np.int32(mem),
                        standard_name=oemplotlib.DEFAULT_CUBE_MEMBER_COORD,
                    )
                )
                out_cubes.append(separated)

        return out_cubes.merge()

    ref_coord_vals = list(cube.coord(forecast_ref_coord_name).cells())
    if len(ref_coord_vals) > 1:
        ref_coord_vals = list(set(ref_coord_vals))
        ref_cubes = iris.cube.CubeList(
            [
                cube.extract(
                    iris.Constraint(
                        **{forecast_ref_coord_name: lambda cell: cell.point == p.point}
                    )
                )
                for p in ref_coord_vals
            ]
        )
        for c in ref_cubes:
            ref_coord = c.coord(forecast_ref_coord_name)
            if ref_coord.has_bounds():
                new_bounds = ref_coord.bounds[0, :]
            else:
                new_bounds = None
            new_ref_coord = ref_coord.copy(
                points=ref_coord.points[0:1], bounds=new_bounds
            )
            c.remove_coord(forecast_ref_coord_name)
            c.add_aux_coord(new_ref_coord)
        return ref_cubes.merge()
    return cube


@cube_or_cubelist
def aggregate_to_time(
    cube: iris.cube.Cube,
    aggregator: iris.analysis.Aggregator,
    interval_period_minutes: int,
    lower_bound_offset_minutes: int = 0,
    upper_bound_offset_minutes: int = 0,
    time_coord_name: str = DEFAULT_CUBE_TIME_COORD,
    forecast_period_coord_name: str = DEFAULT_CUBE_FC_PERIOD_COORD,
    forecast_ref_coord_name: str = DEFAULT_CUBE_FC_REF_COORD,
) -> Union[iris.cube.CubeList, iris.cube.Cube]:
    """Aggregates the values of a cube within a sequence of specified time windows.

    Generates a sequence of time windows and then aggregates all values from the cube
    that falls within each window. The first time window will start on the whole
    hour from the earliest time entry in the cube.

    Args:
        cube (iris.cube.Cube): The cube from which the values should be aggregated.
        aggregator (iris.analysis.Aggregator): The iris aggregator to use.
        interval_period_minutes (int): The interval between the time windows.
        lower_bound_offset_minutes (int, optional): The offset (in whole minutes) between the
                                                    'point' of the time window and it's lower bound.
                                                    Defaults to 0.
        upper_bound_offset_minutes (int, optional): The offset (in whole minutes) between the 'point'
                                                    of the time window and it's upper bound.
                                                    Defaults to 0.
        time_coord_name (str, optional): Name of the validity time coordinate.
                                         Defaults to DEFAULT_CUBE_TIME_COORD.
        forecast_period_coord_name (str, optional): Name of the forecast period coordinate.
                                                    Defaults to DEFAULT_CUBE_FC_PERIOD_COORD.
        forecast_ref_coord_name (str, optional): Name of the data time coordinate.
                                                 Defaults to DEFAULT_CUBE_FC_REF_COORD.

    Raises:
        TypeError: Raised if aggregator is not an instance of iris.analysis.Aggregator
        ValueError: Raised if the supplied arguments can not be used to create a valid
                    sequence of time windows.
        ValueError: Raised if the time metadata on the cube is incompatible with
                    accumulating within a window. In particular for cubes with
                    running times or multiple forecast reference times.

    Returns:
        Union[iris.cube.CubeList, iris.cube.Cube]: A cube of values accumulated within the
                                                   specified time windows. The time dimension
                                                   of the cube may not be contiguous, if the
                                                   window does not contain any output values
                                                   it will not be included in the output cube.
                                                   If the input cube is actually a CubeList
                                                   then each Cube in the CubeList will be
                                                   aggregated separately.
    """

    if not isinstance(aggregator, iris.analysis.Aggregator):
        raise TypeError(
            "aggregate_to_time: aggregator is not an instance of iris.analysis.Aggregator"
        )

    if interval_period_minutes > 0 and (60 % interval_period_minutes > 0):
        raise ValueError("interval_period_minutes must divide equally in to an hour")

    if is_running_time(cube, time_coord_name=time_coord_name):
        raise ValueError(
            "aggregate_to_time unable to aggregate cubes with running times"
        )

    try:
        if len(cube.coord(forecast_ref_coord_name).points) > 1:
            LOGGER.debug(
                "snap_to_time found multiple ref times\n%s",
                cube.coord(forecast_ref_coord_name),
            )
            raise ValueError(
                f"Unable to snap cubes with more than one {forecast_ref_coord_name}"
            )
        orrig_ref_time = cube.coord(forecast_ref_coord_name).copy()
    except iris.exceptions.CoordinateNotFoundError:
        orrig_ref_time = None

    # take copy before manipulating cube
    cube = cube.copy()

    # Remove coords that will need to be reset later
    try:
        cube.remove_coord(forecast_ref_coord_name)
    except iris.exceptions.CoordinateNotFoundError:
        pass
    try:
        cube.remove_coord(forecast_period_coord_name)
    except iris.exceptions.CoordinateNotFoundError:
        pass

    time_points = [cell.point for cell in cube.coord(time_coord_name).cells()]
    initial_start_time = time_points[0].replace(minute=0, second=0, microsecond=0)
    # add a bit of time on to the end to make sure the final point doesn't get dropped:
    end_time = time_points[-1] + datetime.timedelta(minutes=interval_period_minutes)

    selected_times = iris.cube.CubeList()
    for select_time, lower_bound, upper_bound in time_window_generator(
        initial_start_time,
        end_time,
        datetime.timedelta(minutes=interval_period_minutes),
        lower_bound_offset=datetime.timedelta(minutes=lower_bound_offset_minutes),
        upper_bound_offset=datetime.timedelta(minutes=upper_bound_offset_minutes),
    ):

        slice = cube.extract(
            iris.Constraint(time=lambda cell: lower_bound < cell.point <= upper_bound)
        )
        if slice is None:
            continue

        if len(slice.coord(time_coord_name).points) < 2:
            LOGGER.warning(
                "aggregate_to_time: slice has less than 2 values, skipping aggregation step"
            )
            # update the metadata so that the cubes will merge
            aggregated_slice = slice.copy()
            aggregator.update_metadata(
                aggregated_slice, aggregated_slice.coord(time_coord_name)
            )
        else:
            aggregated_slice = slice.collapsed(time_coord_name, aggregator)

        def datatime_to_timestamp(dt):
            return aggregated_slice.coord(time_coord_name).units.date2num(dt)

        new_time = aggregated_slice.coord(time_coord_name).copy(
            points=[datatime_to_timestamp(select_time)],
            bounds=[
                (datatime_to_timestamp(lower_bound), datatime_to_timestamp(upper_bound))
            ],
        )
        aggregated_slice.replace_coord(new_time)

        if orrig_ref_time:
            aggregated_slice.add_aux_coord(orrig_ref_time)
            DT = orrig_ref_time.cell(0).point
            forecast_period_points = [
                (c.point - DT).total_seconds() / 3600.0 for c in new_time.cells()
            ]
            forecast_period_lower_bound = [
                (c.bound[0] - DT).total_seconds() / 3600.0 for c in new_time.cells()
            ]
            forecast_period_upper_bound = [
                (c.bound[1] - DT).total_seconds() / 3600.0 for c in new_time.cells()
            ]
            new_forecast_period = iris.coords.AuxCoord(
                forecast_period_points,
                standard_name=forecast_period_coord_name,
                bounds=np.column_stack(
                    (forecast_period_lower_bound, forecast_period_upper_bound)
                ),
                units="hours",
            )
            aggregated_slice.add_aux_coord(
                new_forecast_period, aggregated_slice.coord_dims(new_time)
            )

        selected_times.append(aggregated_slice)
    return selected_times.merge_cube()


@cube_or_cubelist
def snap_to_time(
    cube: iris.cube.Cube,
    minutes_past_hour: int = 0,
    max_window_minutes: int = 30,
    time_coord_name: str = DEFAULT_CUBE_TIME_COORD,
    forecast_period_coord_name: str = DEFAULT_CUBE_FC_PERIOD_COORD,
    forecast_ref_coord_name: str = DEFAULT_CUBE_FC_REF_COORD,
) -> Union[iris.cube.CubeList, iris.cube.Cube]:
    """Set the time values of a cube to the nearest time in sequence.

    This is intended to fix the case where some parameters may be output
    on time steps that do not completely align with a whole hour/minute etc.
    A sequence of times is generated and the nearest entry in the cube
    (within a tolerance) will be used for that time in the sequence.

    Args:
        cube (iris.cube.Cube): The cube whose times should be 'snapped'
        minutes_past_hour (int, optional): The number of minutes past the hour that
                                           should be used as the first time in the sequence.
                                           This must divide evenly into a whole hour.
                                           Defaults to 0.
        max_window_minutes (int, optional): How close to the time in the sequence the time in
                                            the cube must be to be considered a match.
                                            Defaults to 30.
        time_coord_name (str, optional): Name of the validity time coordinate.
                                         Defaults to DEFAULT_CUBE_TIME_COORD.
        forecast_period_coord_name (str, optional): Name of the forecast period coordinate.
                                                    Defaults to DEFAULT_CUBE_FC_PERIOD_COORD.
        forecast_ref_coord_name (str, optional): Name of the data time coordinate.
                                                 Defaults to DEFAULT_CUBE_FC_REF_COORD.

    Raises:
        ValueError: Raised if a valid sequence of times can not be constructed for
                    the given arguments.
        ValueError: Raised if the cube has ore than one data time.

    Returns:
        Union[iris.cube.CubeList, iris.cube.Cube]: A new cube with times from the input cube
                                                   snapped to the specified sequence. Note that
                                                   data may be dropped from the original
                                                   cube if it is not within max_window_minutes
                                                   of a valid time.
                                                   If the input cube is actually a CubeList
                                                   then each Cube in the CubeList will be
                                                   aggregated separately.
    """

    if minutes_past_hour > 30:
        raise ValueError("nearest_minute must be less than 30")
    if minutes_past_hour > 0 and (60 % minutes_past_hour > 0):
        raise ValueError("nearest_minute must divide equally in to an hour")

    if is_running_time(cube, time_coord_name=time_coord_name):
        cube = fix_running_cube_time(cube)
        shift_time = _CoordShift(running_parameter=True)
        shift_fc_period = _CoordShift(running_parameter=True)
    else:
        shift_time = _CoordShift(running_parameter=False)
        shift_fc_period = _CoordShift(running_parameter=False)

    if len(cube.coord(forecast_ref_coord_name).points) > 1:
        LOGGER.debug(
            "snap_to_time found multiple ref times\n%s",
            cube.coord(forecast_ref_coord_name),
        )
        raise ValueError(
            f"Unable to snap cubes with more than one {forecast_ref_coord_name}"
        )

    time_points = [cell.point for cell in cube.coord(time_coord_name).cells()]

    # generate sequence of time starting on the first hour

    start_time = time_points[0].replace(minute=0, second=0, microsecond=0)
    end_time = time_points[-1]
    if minutes_past_hour == 0:
        interval = datetime.timedelta(hours=1)
        window_size = datetime.timedelta(minutes=min(max_window_minutes, 30))
    else:
        interval = datetime.timedelta(minutes=minutes_past_hour)
        window_size = datetime.timedelta(
            minutes=min(max_window_minutes, minutes_past_hour / 2.0)
        )

    selected_times = iris.cube.CubeList()

    for select_time, start_time, end_time in time_window_generator(
        start_time,
        end_time,
        interval,
        lower_bound_offset=window_size,
        upper_bound_offset=window_size,
    ):

        nearest_diff = None
        nearest_time = None

        time_cube = cube.extract(
            iris.Constraint(time=lambda cell: start_time <= cell.point < end_time)
        )
        if time_cube is None:
            continue
        for time_slice in time_cube.slices_over(time_coord_name):
            slice_time = time_slice.coord(time_coord_name).cell(0).point

            diff = (slice_time - select_time).total_seconds()
            if nearest_diff is None or abs(diff) < abs(nearest_diff):
                nearest_diff = diff
                nearest_time = time_slice

        if nearest_time:

            snapped = nearest_time.copy()
            snapped_time = snapped.coord(time_coord_name)

            shift = -datetime.timedelta(seconds=nearest_diff)

            snapped.replace_coord(shift_time.shift(snapped_time, shift))

            try:
                snapped_ref_time = snapped.coord(forecast_period_coord_name)
            except iris.exceptions.CoordinateNotFoundError:
                pass
            else:
                snapped.replace_coord(
                    shift_fc_period.shift(
                        snapped_ref_time, shift.total_seconds() / 3600.0
                    )
                )

            selected_times.append(snapped)
    return selected_times.merge_cube()


@cube_or_cubelist
def running_accum_to_period(
    cube: iris.cube.Cube,
    period_minutes: int = 60,
    out_cube_name: str = None,
    time_coord_name: str = DEFAULT_CUBE_TIME_COORD,
    forecast_period_coord_name: str = DEFAULT_CUBE_FC_PERIOD_COORD,
):
    if not times_in_order(cube, time_coord_name=time_coord_name):
        raise ValueError("Time coordinate is out of order")
    period_delta = datetime.timedelta(minutes=period_minutes)
    cube = cube.copy()
    accum_slices = iris.cube.CubeList()

    orrig_cell_method = cube.cell_methods
    if len(orrig_cell_method) != 1:
        raise ValueError("Cube should have one cell method")
    orrig_cell_method = orrig_cell_method[0]

    if period_delta.total_seconds() % 3600 == 0:
        method_period = "{} hour".format(period_delta.total_seconds() // 3600)
    else:
        method_period = "{} minutes".format(period_delta.total_seconds() // 60)
    cell_method = iris.coords.CellMethod(
        method=orrig_cell_method.method,
        coords=time_coord_name,
        intervals=method_period,
    )

    for timeslice in cube.slices_over(time_coord_name):
        time = timeslice.coord(time_coord_name).cell(0)
        last_slice = cube.extract(
            iris.Constraint(time=lambda cell: cell.point == (time.point - period_delta))
        )
        if not last_slice:
            if time.bound[0] == (time.point - period_delta):
                # first slice already accumulated over
                # required period
                last_slice = timeslice
                diff = timeslice.copy()
                diff.cell_methods = None
                diff.rename(None)
            else:
                continue
        else:
            diff = timeslice - last_slice

            diff_time = timeslice.coord(time_coord_name).copy()
            last_slice_coord = last_slice.coord(time_coord_name)
            diff_time.bounds[:, 0] = last_slice_coord.bounds[:, 1]
            diff.add_aux_coord(diff_time)

            diff_time = timeslice.coord(forecast_period_coord_name).copy()

            last_slice_coord = last_slice.coord(forecast_period_coord_name)
            diff_time.bounds[:, 0] = last_slice_coord.bounds[:, 1]
            diff.add_aux_coord(diff_time)

        diff.add_cell_method(cell_method)
        diff.attributes.update(timeslice.attributes)

        accum_slices.append(diff)
        prev_slice = timeslice

    out_cube = accum_slices.merge_cube()

    if out_cube_name:
        out_cube.rename(out_cube_name)

    return out_cube


@cube_or_cubelist
def hrly_accum_to_period(
    cube: iris.cube.Cube,
    period_hours: int = 6,
    out_cube_name: str = None,
    mask_method="all",
    time_coord_name: str = DEFAULT_CUBE_TIME_COORD,
    forecast_period_coord_name: str = DEFAULT_CUBE_FC_PERIOD_COORD,
    forecast_ref_coord_name: str = DEFAULT_CUBE_FC_REF_COORD,
):
    """Sum individual hour periods to longer accumulation periods

    Args:
        cube (iris.cube.Cube): A cube containing hour long accumulations
        period_hours (int, optional): The longer period to which the individual hours should be summed.
                                      Defaults to 6.
        out_cube_name (str, optional): If supplied the returned cubes will be renamed to this value.
                                       Defaults to None in which case the name will be determined automatically.
        mask_method (str, optional): How masked arrays should be handled. 'all' replicates the behaviour of
                                     numpy.ma.sum where any unmasked value in the elements being summed
                                     will result in an unmasked element in the output. 'any' will result in the
                                     output being masked where any of the input elements are.
                                     Defaults to 'all'.
        time_coord_name (str, optional): The name of the time coordinate.
                                         Defaults to DEFAULT_CUBE_TIME_COORD.
        forecast_period_coord_name (str, optional): Name of the forecast period coordinate.
                                                    Defaults to DEFAULT_CUBE_FC_PERIOD_COORD.
        forecast_ref_coord_name (str, optional): The name of the data time coordinate.
                                                 Defaults to DEFAULT_CUBE_FC_REF_COORD.

    Returns:
        Union[iris.cube.CubeList, iris.cube.Cube]: A Cube or CubeList with Cubes summed to 
                                                   period_hours accumulations.
    """

    if mask_method not in ["any", "all"]:
        raise ValueError("hrly_accum_to_period: mask method must be 'any' or 'all'")

    assert not is_running_time(
        cube, time_coord_name=time_coord_name
    ), "hrly_accum_to_period can not work with running accumulations"
    assert isinstance(period_hours, int), "period_hours must be an integer"

    cube = cube.copy()
    cube.coord(forecast_period_coord_name).convert_units("hours")

    assert cube.coord(
        time_coord_name
    ).is_contiguous(), "hrly_accum_to_period: time coord is not contiguous"
    for slice in cube.slices_over(time_coord_name):
        point = slice.coord(time_coord_name).cell(0).point
        bound = slice.coord(time_coord_name).cell(0).bound
        assert (
            bound[1] == point
        ), "hrly_accum_to_period: time upper bound is not equal to the point"
        assert (bound[1] - bound[0]) == datetime.timedelta(
            hours=1
        ), "hrly_accum_to_period: cube has non-hourly time slice"

    # Note: we can't use iris' cube.rolling_window to sum up the time slices
    #       here as the the window is centred and we want accumulations at the
    #       end of the period

    slices_for_add = iris.cube.CubeList()
    summed_slices = iris.cube.CubeList()
    for slice in cube.slices_over(time_coord_name):
        slices_for_add.append(slice)
        if len(slices_for_add) > period_hours:
            slices_for_add = slices_for_add[1:]
        if len(slices_for_add) == period_hours:
            # first merge to a single cube and do the actual sum
            merged = slices_for_add.merge_cube()
            if mask_method == "all" or not isinstance(merged.data, np.ma.MaskedArray):
                # standard numpy method, just use iris.analysis.sum
                sum = merged.collapsed(time_coord_name, iris.analysis.SUM)
            else:
                # mask method should be 'any'
                # Any masked input element should result in a masked output element

                # separate data and mask and sum them separately
                sum_data_cube = merged.copy(data=merged.data.data).collapsed(
                    time_coord_name, iris.analysis.SUM
                )
                sum_mask_cube = merged.copy(
                    data=np.where(merged.data.mask, 1, 0).astype(np.int32)
                ).collapsed(time_coord_name, iris.analysis.SUM)
                # convert mask back to boolean and combine with data
                # to make the new masked array in a sum cube
                sum = sum_data_cube.copy(
                    data=np.ma.MaskedArray(
                        data=sum_data_cube.data,
                        mask=np.where(sum_mask_cube.data > 0, True, False),
                    )
                )

            # now make sure the various time coordinates are correct
            sum.coord(time_coord_name).points = sum.coord(time_coord_name).bounds[:, 1]
            new_fp_bounds = (
                slices_for_add[-1].coord(forecast_period_coord_name).bounds.copy()
            )
            new_fp_bounds[:, 0] = new_fp_bounds[:, 1] - period_hours
            sum.coord(forecast_period_coord_name).bounds = new_fp_bounds
            sum.coord(forecast_period_coord_name).points = sum.coord(
                forecast_period_coord_name
            ).bounds[:, 1]
            sum.replace_coord(slices_for_add[-1].coord(forecast_ref_coord_name))
            if out_cube_name:
                sum.rename(out_cube_name)
            else:
                sum.rename(f"{period_hours} hourly accumulations of {sum.name()}")
            summed_slices.append(sum)

    return summed_slices.merge_cube()


def masked_cube_slicer(
    cube: iris.cube.Cube, *args, **kwargs
) -> Generator[iris.cube.Cube, None, None]:
    """Slices over cubes with masked data retaining the mask

    There is a bug in some versions of iris that means the mask on cube data
    can be dropped when slicing over a cube. This function creates slices
    of a cube where the mask is correctly retained.

    .. warning ::
       The data will be realised before the cube is sliced

    Args:
        cube (iris.cube.Cube): The iris cube for which slices should be generated
        slice_method (Union[Literal["slices"], Literal["slices_over"]]):
                               The cube slicing method to use. Defaults to 'slices'.
        *args: Arguments to be passed in to the cube slicing method.
        **kwargs: Keyword arguments to be passed in to the cube slicing method.

    Yields:
        Generator[iris.cube.Cube]: A cube sliced as generated by the cube slicing
                                   method but with an intact mask.
    """

    slice_method = kwargs.pop("slice_method", "slices")
    if slice_method not in ["slices", "slices_over"]:
        raise ValueError(
            "masked_cube_slicer: slice method must be 'slices' or 'slices_over'"
        )
    if not np.ma.isMaskedArray(cube.data):
        raise ValueError("masked_cube_slicer: cube.data is not a MaskedArray")

    data_cube = cube.copy(data=cube.data.data.copy())
    # use getmaskarray to make sure the data in mask_cube is the same
    # shape as the data in data_cube
    mask_cube = cube.copy(data=np.ma.getmaskarray(cube.data).copy())

    cube_iterator = zip(
        getattr(data_cube, slice_method)(*args, **kwargs),
        getattr(mask_cube, slice_method)(*args, **kwargs),
    )
    for cube_slice_data, cube_slice_mask in cube_iterator:
        yield cube_slice_data.copy(
            data=np.ma.MaskedArray(data=cube_slice_data.data, mask=cube_slice_mask.data)
        )


def masked_cubelist_merger(cubelist: iris.cube.CubeList) -> iris.cube.Cube:
    """Merges a CubeList of Cubes with masked data into a single Cube

    There is a bug in some versions of iris that means merging a CubeList
    will drop the masks from the Cubes. This function will return a Cube
    with the mask intact.

    .. note ::
       This is the equivalent of using the 'merge_cube' method on a
       normal iris.cube.CubeList

    .. warning ::
       The data will be realised before the CubeList is merged

    Args:
        cubelist (iris.cube.CubeList): The CubeList to be merged

    Returns:
        iris.cube.Cube: The Cube created by merging all of the cubes
                        in cubelist
    """

    if not isinstance(cubelist, iris.cube.CubeList):
        raise ValueError(
            "masked_cubelist_merger: cubelist must be an instance of iris.cube.CubeList"
        )

    cube_data_list = iris.cube.CubeList()
    cube_mask_list = iris.cube.CubeList()
    for cube in cubelist:
        if not np.ma.isMaskedArray(cube.data):
            raise ValueError(
                "masked_cubelist_merger: cubelist contains a cube "
                "whose data is not a MaskedArray"
            )
        cube_data_list.append(cube.copy(data=cube.data.data.copy()))
        cube_mask_list.append(cube.copy(data=np.ma.getmaskarray(cube.data).copy()))

    cube_data = cube_data_list.merge_cube()
    cube_mask = cube_mask_list.merge_cube()

    return cube_data.copy(
        data=np.ma.MaskedArray(data=cube_data.data, mask=cube_mask.data)
    )


NANMIN_AGGREGATOR = iris.analysis.Aggregator("nanmin", np.nanmin)
"""An :class:`iris.analysis.Aggregator` instance that calculates
   the minimum over a :class:`iris.cube.Cube`, as computed by
   :func:`numpy.nanmin`.
   
   Unlike the standard `iris.analysis.MIN` aggregator this will
   return the a minimum value for an element even if some of the
   inputs are invalid (e.g. NaN). If all inputs are NaN the output
   will be NaN.

   .. note ::
      This does not handle masked arrays correctly. Consider filling
      masked elements with NaN before using this aggregator.
"""

NANSUM_AGGREGATOR = iris.analysis.Aggregator("nansum", np.nansum)
"""An :class:`iris.analysis.Aggregator` instance that calculates
   the sum over a :class:`iris.cube.Cube`, as computed by
   :func:`numpy.nansum`.
   
   Unlike the standard `iris.analysis.SUM` aggregator this will
   return the a sum for an element even if some of the
   inputs are invalid (e.g. NaN).
   
   In NumPy versions <= 1.9.0 Nan is returned for slices that are all-NaN or empty.
   In later versions zero is returned.

   .. note ::
      This does not handle masked arrays correctly. Consider filling
      masked elements with NaN before using this aggregator.
"""
