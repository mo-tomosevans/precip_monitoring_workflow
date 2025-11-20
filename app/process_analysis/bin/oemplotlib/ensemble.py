from typing import Union

from oemplotlib import LOGGER as _OEMLOGGER
import oemplotlib
import iris

LOGGER = _OEMLOGGER.getChild(__name__)


class CubeLaggingError(RuntimeError):
    """Error raised if a Cube of lagged members could not be created"""


class NotEnsemlbeError(CubeLaggingError):
    "Raised if a cube does not have the correct metadata to describe an ensemble"


class MissingMembersError(CubeLaggingError):
    "Raised if a cube does not have the expected number of ensemble members"


class TimeLagPreprocessor:
    "Time lag a Cube or CubeList"

    def __init__(
        self,
        cube: Union[iris.cube.Cube, iris.cube.CubeList],
        n_expected_mems=None,
        forecast_start=None,
        forecast_end=None,
        regridder=None,
    ) -> None:
        """Converts a Cube or Cubelist to a time lagged ensemble

        Args:
            cube (Union[iris.cube.Cube, iris.cube.CubeList]): Cube or CubeList to be lagged
            n_expected_mems (int): Expected number of members in the final ensemble.
            forecast_start (datetime like): Start datetime of the ensemble.
            forecast_end (datetime like, optional): End datetime of the ensemble. Defaults to None.
            regridder (oemplotlib.gridtools.SimpleRegridder, optional):
                                          Regridder to use to change the grid of the output
                                          ensemble. Defaults to None.

        Raises:
            ValueError: Raised if forecast_start or n_expected_mems are not set
        """

        self._initial_cube = self._tidy_cube_times(cube)

        self._current_members_cube = {}
        self._time_windowed_cube = None
        self._cube_lagged = None

        if forecast_start is None:
            raise ValueError("forecast start must be specified")

        self._forecast_start = forecast_start
        self._forecast_end = forecast_end

        try:
            self._n_expected_mems = int(n_expected_mems)
        except Exception:
            self._n_expected_mems = None
        if self._n_expected_mems is None:
            raise ValueError("n_expected mems must be set")

        self._n_expected_mems = n_expected_mems

        self._regridder = regridder

        self._setup_time_window()

    @staticmethod
    def _tidy_cube_times(cube):

        if isinstance(cube, iris.cube.CubeList):
            # problematic time coord has probably already been fixed
            cube = iris.cube.CubeList(c.copy() for c in cube)
            cube = oemplotlib.cube_utils.snap_to_time(
                cube, minutes_past_hour=0, max_window_minutes=5,
            )
        else:
            # a single cube of multiple ensemble members is likely
            # to have a multidimensional coordinate that needs to be split
            cube = oemplotlib.cube_utils.snap_to_time(
                oemplotlib.cube_utils.separate_realization_time(cube.copy()),
                minutes_past_hour=0,
                max_window_minutes=5,
            )
        return cube

    def _setup_time_window(self):

        cube = self._initial_cube

        try:
            if isinstance(cube, iris.cube.CubeList):
                [c.coord(oemplotlib.DEFAULT_CUBE_MEMBER_COORD) for c in cube]
            else:
                cube.coord(oemplotlib.DEFAULT_CUBE_MEMBER_COORD)
        except iris.iris.exceptions.CoordinateNotFoundError:
            LOGGER.warning(
                "%s unable to find %s coordinate, assuming cube is deterministic",
                self.__class__.__name__,
                oemplotlib.DEFAULT_CUBE_MEMBER_COORD,
            )
        else:
            cube = oemplotlib.cube_utils.extract_common_member_times(cube)

        if self._forecast_end:
            cube = oemplotlib.cube_utils.extract_time_window(
                cube, self._forecast_start, self._forecast_end,
            )

        if self._regridder:
            if isinstance(cube, iris.cube.CubeList):
                cube = iris.cube.CubeList(self._regridder.regrid(c) for c in cube)
            else:
                cube = self._regridder.regrid(cube)

        self._time_windowed_cube = cube

    def get_start_cycle_members(self, period="overlapping",) -> iris.cube.Cube:
        """Return a Cube containing only the members from the current cycle

        Returns all members from the forecast cycle specified by "forecast_start"
        when initialising the ProbCubePreprocessor

        Args:
            period (Literal["overlapping", "overlapping_and_future", "all"]):
                The time period over which the members should be extracted
                * 'overlapping' - only times where all members are available
                * 'overlapping_and_future' - remove times from the start of the
                  forecast where not all members are available
                * 'all' - all times for members from the current cycle

        Returns:
            iris.cube.Cube: Cube containing members from the current cycle.
        """

        if period not in ["overlapping", "overlapping_and_future", "all"]:
            raise ValueError("get_start_cycle_members: invalid period specified")

        if period in self._current_members_cube:
            return self._current_members_cube[period]

        # extract members produced on the current cycle
        realizations_cube = oemplotlib.cube_utils.extract_time_window(
            self._time_windowed_cube,
            self._forecast_start,
            self._forecast_start,
            time_coord_name=oemplotlib.DEFAULT_CUBE_FC_REF_COORD,
        )

        if isinstance(realizations_cube, iris.cube.CubeList):
            realizations_cube = realizations_cube.merge_cube()

        if period == "overlapping":
            # we only need the overlapping period
            current_members_cube = realizations_cube
        else:

            try:
                if isinstance(realizations_cube, iris.cube.CubeList):
                    [
                        c.coord(oemplotlib.DEFAULT_CUBE_MEMBER_COORD)
                        for c in realizations_cube
                    ]
                else:
                    realizations_cube.coord(oemplotlib.DEFAULT_CUBE_MEMBER_COORD)
            except iris.iris.exceptions.CoordinateNotFoundError:
                LOGGER.warning(
                    "%s unable to find %s coordinate, assuming cube is deterministic",
                    self.__class__.__name__,
                    oemplotlib.DEFAULT_CUBE_MEMBER_COORD,
                )
                current_members_cube = self._initial_cube
            else:
                constraint = iris.Constraint(
                    **{
                        oemplotlib.DEFAULT_CUBE_MEMBER_COORD: lambda cell: cell.point
                        in realizations_cube.coord(
                            oemplotlib.DEFAULT_CUBE_MEMBER_COORD
                        ).points
                    }
                )
                current_members_cube = self._initial_cube.extract(constraint)

            if period == "overlapping_and_future":
                current_members_cube = oemplotlib.cube_utils.extract_time_window(
                    current_members_cube,
                    self._forecast_start,
                    None,
                    time_coord_name=oemplotlib.DEFAULT_CUBE_TIME_COORD,
                )

            # else period == 'all', no further filtering required

            if isinstance(current_members_cube, iris.cube.CubeList):
                current_members_cube = current_members_cube.merge_cube()

        # All of the above metadata manipulation can result in time being an
        # aux coord in which case moxie won't slice it properly. Attempt
        # to promote it to a dim coord as a precaution.
        try:
            iris.util.promote_aux_coord_to_dim_coord(
                current_members_cube, oemplotlib.DEFAULT_CUBE_TIME_COORD
            )
        except ValueError:
            if (
                len(
                    current_members_cube.coord(
                        oemplotlib.DEFAULT_CUBE_TIME_COORD
                    ).points
                )
                == 1
            ):
                LOGGER.exception(
                    "Unable to convert single valued %s coordinate to DimCoord, attempting to continue, cube was\n%s",
                    oemplotlib.DEFAULT_CUBE_TIME_COORD,
                    current_members_cube,
                )
            else:
                raise

        self._current_members_cube[period] = current_members_cube
        return current_members_cube

    def get_lagged_cube(self):
        """Time lag a number of cubes in to a single ensemlbe.

        Adjusts the start time and lead time of all of the cubes
        so that they are the same as the current cycle which is
        specified by "forecast_start" when initialising the ProbCubePreprocessor
         

        Raises:
            NotEnsemlbeError: Raised if the class member current_members_cube
                              does not have the metadata expected from an ensemble.
            MissingMembersError: Raised if the ensemble does not have the expected number
                                 of members as set by n_expected_mems
            CubeLaggingError: Raised if an unknown error happened when trying
                              to create a lagged cube

        Returns:
            iris.cube.Cube: A time lagged cube of the ensemble.
        """

        if self._cube_lagged:
            return self._cube_lagged

        cube = self._time_windowed_cube

        try:
            if isinstance(cube, iris.cube.CubeList):
                cube[0].coord(oemplotlib.DEFAULT_CUBE_MEMBER_COORD)
            else:
                cube.coord(oemplotlib.DEFAULT_CUBE_MEMBER_COORD)
        except iris.exceptions.CoordinateNotFoundError as excp:
            raise NotEnsemlbeError(
                f"Unable to find {oemplotlib.DEFAULT_CUBE_MEMBER_COORD} coordinate"
            ) from excp
        else:
            if isinstance(cube, iris.cube.CubeList):
                # If everything up to this point has worked the cube list
                # should have a cube for each data time with the realizations
                # from that data time
                cube_lagged = iris.cube.CubeList()
                for c in cube:
                    # copy each cube individually as Cubelist.copy only
                    # makes a shallow copy
                    lag = c.copy()
                    lag.remove_coord(oemplotlib.DEFAULT_CUBE_FC_PERIOD_COORD)
                    lag.remove_coord(oemplotlib.DEFAULT_CUBE_FC_REF_COORD)

                    # iris can't concatenate cubes with gaps in realization number,
                    # i.e. a cube with realizations 0, 18, 19 won't concatenate with
                    # one with 15, 16, 17. So split in to individual members before concatenating.
                    for mslice in lag.slices_over(oemplotlib.DEFAULT_CUBE_MEMBER_COORD):
                        cube_lagged.append(mslice)

                cube_lagged = cube_lagged.merge_cube()
            else:
                cube_lagged = cube.copy()
                cube_lagged.remove_coord(oemplotlib.DEFAULT_CUBE_FC_PERIOD_COORD)
                cube_lagged.remove_coord(oemplotlib.DEFAULT_CUBE_FC_REF_COORD)

            n_members_found = len(
                cube_lagged.coord(oemplotlib.DEFAULT_CUBE_MEMBER_COORD).points
            )
            if n_members_found != self._n_expected_mems:
                raise MissingMembersError(
                    f"cube has {n_members_found} members but {self._n_expected_mems} are expected:\n{cube}"
                )

            cube_lagged.add_aux_coord(
                self.get_start_cycle_members().coord("forecast_reference_time")
            )

            cube_lagged = oemplotlib.cube_utils.fctime_from_datatime(cube_lagged)

            # All of the above metadata manipulation can result in time being an
            # aux coord in which case moxie won't slice it properly. Attempt
            # to promote it to a dim coord as a precaution.
            iris.util.promote_aux_coord_to_dim_coord(
                cube_lagged, oemplotlib.DEFAULT_CUBE_TIME_COORD
            )

            self._cube_lagged = cube_lagged
            return cube_lagged

        raise CubeLaggingError("Unknown Error trying to make Lagged Cube")
