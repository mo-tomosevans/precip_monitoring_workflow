from typing import List, Union
from abc import ABC, abstractmethod
import configobj
from pathlib import Path
import iris
import numpy as np
from cf_units import Unit
import oemplotlib
from oemplotlib.utils import get_lat_lon_from_cube

GRID_POINT_OUTPUT_FMT = oemplotlib.CONFIG["grid_settings"]["output format"]
# appropriate default value for most model grids
DEFAULT_WIND_ARROW_SPACING = 20

LOGGER = oemplotlib.LOGGER.getChild(__name__)


class GridCubeHandler:
    """Convenience class for handling plotting on different grids"""

    def __init__(self, grid: Union[iris.cube.Cube, None]) -> None:
        """Initialise Grid cube

        Args:
            grid (Union[iris.cube.Cube, None]): A cube containing grid information.
                                                The Latitude and Longitude dimensions should
                                                define the grid and the cube's name should be
                                                the name of the grid.

                                                If the grid is set to None all of this classes
                                                methods will exit without raising errors, returning
                                                None where a return variable is required.
        """
        self._grid = grid
        if self._grid is None:
            LOGGER.info(
                "GridSettings initialized with None, all grid settings will skipped"
            )
            self._projection = None
            self._crs = None
            self._domain = None
            self._plotting_projection = None
            self._plotting_domain = None
            self._wind_arrow_spacing = None
        else:
            self._projection = self._grid.coord_system().as_cartopy_projection()
            self._crs = self._grid.coord_system().as_cartopy_crs()
            self._domain = self.domain_from_cube(self._grid)

            if self._grid.attributes:
                self._plotting_projection = self._grid.attributes.get(
                    "plotting_projection", None
                )
                self._plotting_domain = self._grid.attributes.get(
                    "plotting_domain", None
                )
                self._wind_arrow_spacing = self._grid.attributes.get(
                    "wind_arrow_spacing", None
                )
                if not (
                    (self._plotting_projection and self._plotting_domain)
                    or (
                        self._plotting_projection is None
                        and self._plotting_domain is None
                    )
                ):
                    raise ValueError(
                        "A grid definition may contain either both 'plotting_projection' and 'plotting_extent' or neither"
                    )
            else:
                self._plotting_projection = None
                self._plotting_domain = None
                self._wind_arrow_spacing = None

    @staticmethod
    def domain_from_cube(cube: iris.cube.Cube) -> List[float]:
        """Extract the domain covered by a cube.

        This function extracts the domain from a cube with the assumption that the
        cube has latitude and longitude coordinates and that the first value
        in the coordinate is the minimum value and the last is the maximum.

        Args:
            cube (iris.cube.Cube): A cube containing longitude and latitude like
                                   coordinates.

        Returns:
            List[float]: The domain defined as
                         [minimum_longitude, maximum_longitude, minimum_latitude, maximum_latitude]
        """

        # Note, moxie has a function for extracting this, but as this
        # module is intended to work in environments where moxie may not
        # be available, we write our own simplified version.

        lat, lon = oemplotlib.utils.get_lat_lon_from_cube(cube)
        if lat.bounds is None:
            minlat = lat.points[0]
            maxlat = lat.points[-1]
        else:
            minlat = lat.bounds[0][0]
            maxlat = lat.bounds[-1][1]
        if lon.bounds is None:
            minlon = lon.points[0]
            maxlon = lon.points[-1]
        else:
            minlon = lon.bounds[0][0]
            maxlon = lon.bounds[-1][1]
        return [minlon, maxlon, minlat, maxlat]

    def setup_cubes_attributes(self, *args: iris.cube.Cube) -> None:
        """Add grid information to cube attributes dictionaries (in place)

        Given a cube or list of cubes, this function adds grid information to
        their attribute dictionary:
        * grid: the name of the grid

        Raises:
            ValueError: If one of the arguments supplied is not a cube
        """
        if self._grid:
            for cube in args:
                if not isinstance(cube, iris.cube.Cube):
                    raise ValueError("setup_cubes argument is not a cube")
            gridname = self._grid.name()
            for cube in args:
                cube.attributes["grid"] = gridname

    @property
    def native_projection(self):
        """The native projection of the grid

        The projection that would be plotted by cartopy
        if a cube on this grid was supplied without a projection argument
        """
        if self._grid:
            return self._projection
        return None

    @property
    def native_crs(self):
        """The native coordinate reference system of the grid

        The cartopy crs equivalent of the coordinate system on which
        this gird is defined.
        """

        if self._grid:
            return self._crs
        return None

    @property
    def native_domain(self):
        """The domain of the grid.

        The domain of the grid specified as [x_min, x_max, y_min, y_max]
        in the coordinate system specified by native_crs
        """
        if self._grid:
            return self._domain
        return None

    @property
    def plotting_domain(self):
        """The projection domain for the grid.

        The projection domain for the grid defined in the crs
        obtained from plotting_projection
        """

        if self._plotting_domain:
            return self._plotting_domain
        return self.native_domain

    @property
    def plotting_projection(self):
        """The projection of the grid

        The projection used by cartopy when making plots.
        This defaults to 'native_projection' when the grid does not define an explicit projection.
        """
        if self._plotting_projection:
            return self._plotting_projection
        return self.native_projection


class GridfileManager:
    """A class to read and write grid conf files"""

    def __init__(self):
        self._hgrid_coords = [
            ["latitude", "longitude"],
            ["grid_latitude", "grid_longitude"],
        ]
        self._coord_systems = {
            "GeogCS": [
                "semi_major_axis",
                "semi_minor_axis",
                "longitude_of_prime_meridian",
            ],
            "RotatedGeogCS": [
                "grid_north_pole_latitude",
                "grid_north_pole_longitude",
                "north_pole_grid_longitude",
                "ellipsoid",
            ],
        }
        self._dimcord_kwargs = [
            "points",
            "bounds",
            "standard_name",
            "long_name",
            "var_name",
            "units",
            "coord_system",
            "circular",
        ]

    def conf_from_cube(self, cube: iris.cube.Cube) -> str:
        """Extracts the grid information from an iris cube and writes it to a sring.

        Given a cube, the horzontal grid coordinates are extracted, and then converted
        to configobj ini style config. This should contain enough information
        to reconstruct a dummy cube representing the same horizontal grid.

        Args:
            cube (iris.cube.Cube): The iris cube to be converted

        Raises:
            ValueError: Raised if a horzontal grid can not be extracted
                        from the cube.

        Returns:
            str: The grid config as a string
        """

        out_dict = {}

        for hgrid_coords in self._hgrid_coords:
            try:
                for _ in cube.slices(hgrid_coords):
                    # first slice should have a suitable grid
                    break
            except iris.exceptions.CoordinateNotFoundError:
                # try the next possible_hgrid_coords
                continue
            # found the the hgrid_coords
            break
        else:
            # none of the possible hgrid_coords worked
            raise ValueError("Unable to find a grid in the given cube")

        for coord_name in hgrid_coords:

            cube_coord = cube.coord(coord_name)
            out_dict[cube_coord.name()] = self._get_dim_coord(cube_coord)

        config = configobj.ConfigObj(out_dict, unrepr=True)
        return "\n".join(config.write())

    def _get_dim_coord(self, coord):
        coord_dict = {}
        for coord_kwarg in self._dimcord_kwargs:
            if coord_kwarg == "points":
                coord_dict["points"] = [
                    GRID_POINT_OUTPUT_FMT.format(p) for p in coord.points
                ]
            elif coord_kwarg == "bounds":
                if coord.has_bounds():
                    coord_dict[coord_kwarg] = [
                        [
                            GRID_POINT_OUTPUT_FMT.format(p[0]),
                            GRID_POINT_OUTPUT_FMT.format(p[1]),
                        ]
                        for p in coord.bounds
                    ]
            elif coord_kwarg == "coord_system":
                coord_dict[coord_kwarg] = self._get_coord_sys(coord)
            elif coord_kwarg == "units":
                coord_dict[coord_kwarg] = str(getattr(coord, coord_kwarg))
            else:
                coord_dict[coord_kwarg] = getattr(coord, coord_kwarg)
        return coord_dict

    def _get_coord_sys(self, coord):

        grid_dict = {}

        cube_coord_sys = coord.coord_system

        for coord_sys_name in self._coord_systems.keys():
            # use type rather than isinstance because we can't handle subclasses
            if type(cube_coord_sys) == getattr(iris.coord_systems, coord_sys_name):
                grid_dict["type"] = coord_sys_name
                break
        else:
            raise ValueError(f"Unable to handle {type(cube_coord_sys)}")

        for coord_variable in self._coord_systems[coord_sys_name]:
            if coord_variable == "ellipsoid":
                ellipsoid = cube_coord_sys.ellipsoid
                if type(ellipsoid) != iris.coord_systems.GeogCS:
                    raise ValueError("Can only handle ellipsoids of type GeogCS")
                ellipsoid_dict = {"type": "GeogCS"}
                for ellipsoid_variable in self._coord_systems["GeogCS"]:
                    ellipsoid_dict[ellipsoid_variable] = getattr(
                        ellipsoid, ellipsoid_variable
                    )
                grid_dict["ellipsoid"] = ellipsoid_dict
            else:
                grid_dict[coord_variable] = getattr(cube_coord_sys, coord_variable)

        return grid_dict

    def cube_from_conf(self, conffile: str) -> iris.cube.Cube:
        """Reads in a grid file and converts it to an empty iris cube.

        Reads the grid file and uses it to build a cube, with np.NaN data,
        that should be suitable for use in operations that require the definition
        of a grid, for example, regridding.

        Args:
            conffile (str): The path to the grid file to be read in.

        Returns:
            iris.cube.Cube: A cube containing numpy.NaN data but with
                            horizontal coordinates as specified in the grid file.
        """
        config = configobj.ConfigObj(conffile, unrepr=True)

        dim_coords_and_dims = []
        attributes = {}
        np_size = []

        for i, (config_key, coord_kwargs) in enumerate(config.items()):
            if config_key == "plot_settings":
                if "projection_coord_system" in coord_kwargs:
                    attributes["plotting_projection"] = self._build_coord_sys(
                        coord_kwargs["projection_coord_system"]
                    ).as_cartopy_projection()
                if "projection_domain" in coord_kwargs:
                    str_lst = coord_kwargs["projection_domain"]["points"]
                    domain_lst = [float(i) for i in str_lst]
                    attributes["plotting_domain"] = domain_lst
                if "wind_arrow_spacing" in coord_kwargs:
                    attributes["wind_arrow_spacing"] = coord_kwargs[
                        "wind_arrow_spacing"
                    ]
            else:
                # convert special kwargs back to the correct type
                if "units" in coord_kwargs:
                    coord_kwargs["units"] = Unit(coord_kwargs["units"])
                if "coord_system" in coord_kwargs:
                    coord_kwargs["coord_system"] = self._build_coord_sys(
                        coord_kwargs["coord_system"]
                    )
                if "bounds" in coord_kwargs:
                    coord_kwargs["bounds"] = np.array(
                        [[float(p[0]), float(p[1])] for p in coord_kwargs["bounds"]]
                    )
                points = [float(p) for p in coord_kwargs.pop("points")]

                dim_coords_and_dims.append(
                    (iris.coords.DimCoord(points, **coord_kwargs), i)
                )
                np_size.append(len(points))

        data = np.empty(tuple(np_size), np.float32)
        data[:] = np.NaN

        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=dim_coords_and_dims,
            attributes=attributes if attributes else None,
        )

        return cube

    def _build_coord_sys(self, coord_conf):
        kwargs_dict = {}
        coord_type = getattr(iris.coord_systems, coord_conf["type"])
        for key, val in coord_conf.items():
            if key == "ellipsoid":
                kwargs_dict["ellipsoid"] = self._build_coord_sys(val)
            elif key == "type":
                continue
            else:
                kwargs_dict[key] = val
        return coord_type(**kwargs_dict)

    def load_grid(
        self, config_dir: Path, model_name: str, gridfile_name: str = "grid.conf"
    ) -> iris.cube.Cube:
        """Convenience function for loading grid definition files.

        This function assumes that model grid definitions are stored in a
        directory structure: config_dir/model_name/gridfile_name.

        Args:
            config_dir (Path): Top level paths where model configs are stored
            model_name (str): The name of a model. This should be a folder under
                              config_dir.
            gridfile_name (str, optional): The name used for the actual
                                           grid definition file.
                                           Defaults to "grid.conf".

        Returns:
            iris.cube.Cube: A cube containing numpy.NaN data but with
                            horizontal coordinates as specified in the grid file.
        """
        grid_file = config_dir / model_name / gridfile_name
        LOGGER.debug("Loading grid from %s", grid_file)
        gridcube = self.cube_from_conf(str(grid_file))
        gridcube.rename(model_name)
        return gridcube


class OEMRegridderABC(ABC):
    def __init__(
        self,
        source_grid: iris.cube.Cube,
        target_grid: iris.cube.Cube,
        scheme=iris.analysis.Nearest(extrapolation_mode="mask"),
        target_grid_name: str = None,
    ) -> None:
        # Iris bug? gridding can go wrong (offset error?) for grids without bounds
        # even for schemes that don't technically require them. Make copies and
        # ensure out target grid has bounds.

        self._target_grid = target_grid.copy()
        self.scheme = scheme
        self._target_grid_name = target_grid_name if target_grid_name else None

        self._target_grid_handler = GridCubeHandler(target_grid)
        self._wind_arrow_spacing = target_grid.attributes.get(
            "wind_arrow_spacing", DEFAULT_WIND_ARROW_SPACING
        )

    @property
    def target_grid(self):
        return self._target_grid

    @property
    def target_grid_name(self):
        return self._target_grid_name

    @property
    def target_grid_plot_handler(self):
        """A GridCubeHandler to control plotting"""

        return self._target_grid_handler

    @property
    def wind_arrow_spacing(self):
        """Index value for spacing between wind arrows which varies between grids. Defaults to 20."""
        return self._wind_arrow_spacing

    @abstractmethod
    def regrid(self, cube: iris.cube.Cube) -> iris.cube.Cube:
        """Regrid the given cube using the cached regridder

        Args:
            cube (iris.cube.Cube): The cube to be regridded

        Returns:
            iris.cube.Cube: A new cube with the regridded data
        """
        pass


class NullRegridder(OEMRegridderABC):
    """Regridder to fall back to if no 'regridder' is defined, using the original cube's grid"""

    def __init__(
        self,
        source_grid: iris.cube.Cube,
        target_grid_name: str = None,
    ) -> None:
        """
        Set up...

        Args:
            source_grid (iris.cube.Cube): A cube defining the source grid of the data to be regridded.
            target_grid_name (str = None): Name of the target grid cube. Defaults to None.
        """
        lat, lon = oemplotlib.utils.get_lat_lon_from_cube(source_grid)
        for target_grid in source_grid.slices_over([lat.name(), lon.name()]):
            # find the 2d slice that defines the cube's grid
            break

        super().__init__(source_grid, target_grid, None, target_grid_name)

    def regrid(self, cube: iris.cube.Cube) -> iris.cube.Cube:
        copied_cube = cube.copy()
        if self._target_grid_name:
            copied_cube.attributes["grid"] = self._target_grid_name
        return copied_cube


class SimpleRegridder(OEMRegridderABC):
    """Automatically sets up and builds a basic regridder"""

    def __init__(
        self,
        source_grid: iris.cube.Cube,
        target_grid: iris.cube.Cube,
        scheme=iris.analysis.Nearest(extrapolation_mode="mask"),
        target_grid_name: str = None,
    ) -> None:
        """
        Set up...

        Args:
            source_grid (iris.cube.Cube): A cube defining the source grid of the data
                                          to be regridded.
            target_grid (iris.cube.Cube): A cube defining the target grid on to which
                                          data should be regriddded
            scheme (an iris regridding scheme, optional): The scheme to use for regridding.
                                                          Defaults to iris.analysis.Nearest(
                                                              extrapolation_mode="mask"
                                                          ).
            target_grid_name (str, optional): If specified this will be added to the output
                                              cube's attributes dictionary with the key 'grid'. Defaults to None.
        """

        super().__init__(source_grid, target_grid, scheme, target_grid_name)

        lat, lon = get_lat_lon_from_cube(self._target_grid)
        if not lat.has_bounds():
            lat.guess_bounds()
        if not lon.has_bounds():
            lon.guess_bounds()

        self._regridders = [self.scheme.regridder(source_grid, self._target_grid)]

    def regrid(self, cube: iris.cube.Cube) -> iris.cube.Cube:
        """Regrid the given cube using the cached regridder

        Args:
            cube (iris.cube.Cube): The cube to be regridded

        Returns:
            iris.cube.Cube: A new cube with the regridded data
        """
        for regridder in self._regridders:
            try:
                regridded = regridder(cube)
                break
            except ValueError as e:
                if (
                    str(e)
                    == "The given cube is not defined on the same source grid as this regridder."
                ):
                    continue
                else:
                    raise
        else:
            LOGGER.warning(
                "SimpleRegridder: cube on different grid to existing source cube(s) grid(s), "
                "creating new regridder and attempting to continue"
            )
            new_regridder = self.scheme.regridder(cube, self._target_grid)
            regridded = new_regridder(cube)
            self._regridders.append(new_regridder)

        if self._target_grid_name:
            regridded.attributes["grid"] = self._target_grid_name
        return regridded


class MultiStepRegridder(OEMRegridderABC):
    """Regrid a cube using multiple cached regridding steps

    Chain together a series of OEM regridders to regrid a cube in multiple
    steps.

    The call signature to initialise each of the regridders must be
    (source_grid, target_grid, **kwargs)
    """

    def __init__(
        self,
        source_grid: iris.cube.Cube,
        target_grid: iris.cube.Cube,
        regridding_schemes: List = None,
        target_grid_name: str = None,
    ) -> None:
        """
        Initialise the MultiSetpRegridder instance

        Args:
            source_grid (iris.cube.Cube): A cube defining the source grid of the data
                                          to be regridded.
            target_grid (iris.cube.Cube): A cube defining the target grid on to which
                                          data should be regriddded
            regridding_schemes (list, optional): A list defining the regridding steps.
                                                 Each entry in the list must be a tuple whose
                                                 first element must an OEMRegridder class.
                                                 The second element must be the intermediate target grid
                                                 for for the regridding step.
                                                 The final element must be a dictionary with any
                                                 keyword arguments required to set up the regridder,
                                                 it may be empty.
                                                 Defaults to a single regridding step using the
                                                 :py:class:`SimpleRegridder` with its default settings.
            target_grid_name (str, optional): If specified this will be added to the output
                                              cube's attributes dictionary with the key 'grid'. Defaults to None.
        """

        # Call super with scheme set to None
        # to set up basic properties
        super().__init__(
            source_grid,
            target_grid,
            None,
            target_grid_name,
        )
        # Now set up multi-step regridding scheme
        self.scheme = self._configure_regridders(source_grid, regridding_schemes)

    def _configure_regridders(self, source_grid, scheme_list):
        if scheme_list is None:
            return [
                SimpleRegridder(
                    source_grid,
                    self.target_grid,
                    target_grid_name=self.target_grid_name,
                )
            ]
        elif len(scheme_list) < 1:
            raise ValueError("Invalid number of regridding schemes specified")
        else:
            regridders = []

            # set up intermediate regridding steps
            intermediate_source_grid = source_grid
            for scheme, target_grid, scheme_kwargs in scheme_list[:-1]:
                regridders.append(
                    scheme(intermediate_source_grid, target_grid, **scheme_kwargs)
                )
                intermediate_source_grid = target_grid

            # handle the final step on its own as we want to set the target_grid_name
            # at this point
            scheme, target_grid, scheme_kwargs = scheme_list[-1]

            regridders.append(
                scheme(
                    intermediate_source_grid,
                    target_grid,
                    target_grid_name=self.target_grid_name,
                    **scheme_kwargs,
                )
            )

            target_lat_lon = oemplotlib.utils.get_lat_lon_from_cube(target_grid)
            expected_lat_lon = oemplotlib.utils.get_lat_lon_from_cube(self.target_grid)

            if target_lat_lon != expected_lat_lon:
                raise ValueError(
                    "The final regridding step output does not match the target_grid"
                )
            return regridders

    def regrid(self, cube: iris.cube.Cube) -> iris.cube.Cube:
        """Regrid the given cube using the cached regridders

        Args:
            cube (iris.cube.Cube): The cube to be regridded

        Returns:
            iris.cube.Cube: A new cube with the regridded data
        """

        if not self.scheme:
            raise ValueError(f"{self.__name__} regridding schemes not configured")

        regridded = cube
        for regridder in self.scheme:
            regridded = regridder.regrid(regridded)

        return regridded
