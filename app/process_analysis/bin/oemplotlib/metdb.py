import abc
import collections
import datetime
import copy
from typing import Callable
import numpy as np
import cf_units
import metdb
import iris

import oemplotlib
from oemplotlib import REPLICATION_DIM_NAME

# list of element keys that are only relevant to eomplotlib i.e. not used by metdb
OEM_ELEMENT_KEYS = ["common_name", "units"]

SAVE_FILE_SUFFIX = ".nc"
SECONDS_IN_HOUR = 3600
IRIS_TIMESTAMP_UNIT = "hours"

LOGGER = oemplotlib.LOGGER.getChild(__name__)


class ElementBase(collections.abc.MutableMapping, abc.ABC):
    """Abstract Base Class for handling a MetDB Element"""

    def __init__(self, **kwargs) -> None:
        self.element_dict = copy.deepcopy(kwargs)

    def __getitem__(self, key):
        return self.element_dict[key]

    def __setitem__(self, key, value):
        self.element_dict[key] = value

    def __delitem__(self, key):
        del self.element_dict[key]

    def __iter__(self):
        return iter(self.element_dict)

    def __len__(self):
        return len(self.element_dict)

    def __repr__(self):
        return f"{self.__class__} ({self.common_name})"

    def __str__(self):
        return f"{self.common_name} element"

    @property
    def common_name(self):
        """A human readable name for the element"""

        if "common_name" in self.element_dict:
            return self.element_dict["common_name"]
        elif hasattr(self, "_metdb_element_names") and self._metdb_element_names:
            return "_".join(self._metdb_element_names)
        else:
            return "NODEFAULTNAME"

    @property
    def metdb_elemnt_names(self):
        """The name of the element as used by MetDB"""

        return set(self._metdb_element_names) | self.unique_id_element_names

    def configure_units(self, units_dict: dict):
        """Set the units for the element.

        .. note::
           If there is no entry in the units_dict for the element
           the unit will be left unset.


        Args:
            units_dict (dict): A dictionary of the form {element_name: unit}
                               from which the element unit will be set.
        """
        if "units" not in self:
            elname = self.get("element_name", None)
            if elname:
                unit = units_dict.get(elname, None)
                if unit:
                    self["units"] = unit
                else:
                    LOGGER.warning(
                        "configure_units: no unit found for element %s, will be left unset",
                        self["element_name"],
                    )
            else:
                LOGGER.warning(
                    "configure_units: element does not have a name, "
                    "unable to automatically set units, will be left unset",
                )

    @abc.abstractmethod
    def get_time_cell(ob, *args, **kwargs):
        """Get an iris time coordinate cell from the observation.

        Args:
            ob: A MetDB observation from which time information should be constructed

        Returns:
            tuple: A tuple of the form (point, (start_bound, end_bound))
        """

    @abc.abstractmethod
    def valid_ob(self, ob) -> bool:
        """Test if observation is valid for element

        Args:
            ob: Observation to be Tested

        Returns:
            bool: True of ob has necessary entries to make this element
                  else False.
        """

    @abc.abstractproperty
    def unique_id_element_names(self):
        """Check if a unique id has been correctly configured for this element"""

    def get_element_value_from_ob(self, ob):
        """Gets the value matching this element from a MetDB observation"""

        return ob[self["element_name"]]

    def get_extra_static_coords(self, *args, **kwargs):
        """Get extra iris coordinates that are common to all instances of this element

        Returns a list of tuple with coordinates that are required to describe this
        element that are not common to all types of observation.

        The first entry in the tuple will either be an iris.coord.DimCoord or the name of
        one as a string, the second will be a list of iris.coord.AuxCoord corresponding to the
        DimCoord.
        """

        return []


class SimpleUniqueIDElementMixin:
    @property
    def unique_id_element_names(self):
        """The MetDB element names needed to make a unique_id

        Raises:
            KeyError: Raised if this element does not contain
                      the entries required to construct a unique_id
            ValueError: Raised if the entries in this element that should
                        describe how to construct a unique_id are invalid.
            ValueError: [description]

        Returns:
            set: The MetDB elements needed to make the unique_id
        """

        if "unique_id" not in self.element_dict:
            raise KeyError("unique_id not found")
        id_elements = self.element_dict["unique_id"]
        if callable(id_elements):
            return set()
        if isinstance(id_elements, tuple):
            if not callable(id_elements[-1]):
                raise ValueError(
                    "Final value in unique_id tuple should be a "
                    "callback to construct the unique id"
                )
            return set(id_elements[:-1])
        elif isinstance(id_elements, str):
            return set([id_elements])
        raise ValueError("Incorrect unique_id settings")


class SimpleElement(SimpleUniqueIDElementMixin, ElementBase):
    """Class for handling elements that directly extract one observation"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._metdb_element_names = [self.element_dict["element_name"]]

    def get_time_cell(self, ob):
        return (self.element_dict["timestamp_callback"](ob), None)

    def valid_ob(self, ob) -> bool:
        element_name = self._metdb_element_names[0]
        return element_name in ob and not np.ma.is_masked(ob[element_name])


class SimpleReplicatedElement(SimpleUniqueIDElementMixin, ElementBase):
    """Class for handling elements that directly extract one observation with multiple replications"""

    def __init__(self, **kwargs) -> None:
        default_replications = 1
        self._max_replications = kwargs.pop("max_replications", None)
        super().__init__(**kwargs)
        if self._max_replications is None:
            LOGGER.warning(
                "%s.__init__ 'max_replications' not in kwargs for %s, defaulting to%s",
                self.__class__.__name__,
                self.element_dict["element_name"],
                default_replications,
            )

        # the syntax for replicated elements is
        # Tuple(Tuple(<element name strings>), <number of replications>)
        self._metdb_element_names = [
            ((self.element_dict["element_name"],), self._max_replications)
        ]

    @property
    def common_name(self):
        """A human readable name for the element"""

        if "common_name" in self.element_dict:
            return self.element_dict["common_name"]
        elif hasattr(self, "_metdb_element_names") and self._metdb_element_names:
            return "_".join([f"{e[0]}_{str(e[1])}" for e in self._metdb_element_names])
        else:
            return "NODEFAULTNAME"

    def get_time_cell(self, ob):
        return (self.element_dict["timestamp_callback"](ob), None)

    def valid_ob(self, ob) -> bool:
        element_name = self._metdb_element_names[0][0][0]

        if element_name not in ob:
            return False

        obs = ob[element_name]
        masked = np.ma.is_masked(obs) and obs.mask.all()

        return not masked

    def get_element_value_from_ob(self, ob):
        """Gets the value matching this element and pads it to the expected length"""
        raw_ob = ob[self["element_name"]]
        padded = np.ma.MaskedArray(
            np.full(
                shape=self._max_replications, fill_value=np.NaN, dtype=raw_ob.dtype
            ),
            mask=True,
        )
        padded.data[: len(raw_ob.data)] = raw_ob.data
        if np.ma.is_masked(raw_ob):
            padded.mask[: len(raw_ob.mask)] = raw_ob.mask
        else:
            padded.mask[: len(raw_ob.data)] = False

        return padded

    def get_extra_static_coords(self):
        replication = iris.coords.DimCoord(
            np.array(range(self._max_replications)), units="1"
        )
        replication.rename(REPLICATION_DIM_NAME)
        return [(replication, [])]


class PeriodElement(SimpleUniqueIDElementMixin, ElementBase):
    """Class for handling elements that describe an observation over a period.

    This class is intended for observations in MetDB whose time is given
    by three components, an observation time, an offset to the start of the
    observation period (from the observation time as a duration)
    and an offset to the end of the observation period (also as a duration).
    """

    def __init__(
        self,
        offset_start_element=None,
        offset_end_element=None,
        offset_start_units=None,
        offset_end_units=None,
        required_period=None,
        required_period_units=None,
        time_cell_method=None,
        **kwargs,
    ) -> None:
        """Configures the element.

        Args:
            offset_start_element (str, optional): The MetDB element name of the element describing
                                                  the start of the observation period.
                                                  Defaults to None in which case the value will be read
                                                  from the oemplotlib configuration file.
            offset_end_element (str, optional): The MetDB element name of the element describing
                                                the end of the observation period.
                                                Defaults to None in which case the value will be read
                                                from the oemplotlib configuration file.
            offset_start_units (str, optional): The units to be applied to the offset_start_element.
                                                Defaults to None in which case the value will be read
                                                from the oemplotlib configuration file.
            offset_end_units (str, optional): The units to be applied to the offset_end_element.
                                              Defaults to None in which case the value will be read
                                              from the oemplotlib configuration file.
            required_period ([int, float], optional): The period for over which the observation should be valid.
                                                      Defaults to None in which case the value will be read
                                                      from the oemplotlib configuration file .
            required_period_units (str, optional): The units in which required_period is given.
                                                   Defaults to None in which case the value will be read
                                                   from the oemplotlib configuration file.
            time_cell_method ([dict, iris.coords.CellMethod], optional):
                                                    An iris time cell method to be applied to the cube or a dictionary
                                                    describing how to construct one of the form
                                                    {method:'method name', **cell_method_kwargs}
                                                    where method name is the positional argument used
                                                    to construct a CellMethod and cell_method_kwargs
                                                    are the key word arguments as specified in the iris documentation.
                                                    Defaults to None in which case the value will be read
                                                    from the oemplotlib configuration file.

        Raises:
            TypeError: Raised if time_cell_method is of an incorrect type.
        """
        super().__init__(**kwargs)
        self._metdb_element_names = [self.element_dict["element_name"]]
        if offset_start_element:
            self.offset_start_element = offset_start_element
            self._metdb_element_names.append(offset_start_element)
        else:
            self.offset_start_element = None
        if offset_end_element:
            self._metdb_element_names.append(offset_end_element)
            self.offset_end_element = offset_end_element
        else:
            self.offset_end_element = None
        self.offset_start_units = offset_start_units
        self.offset_end_units = offset_end_units
        self.required_period = required_period
        if required_period_units:
            self.required_period_units = required_period_units
        else:
            LOGGER.warning(
                "required_period_units not set for %s, "
                "will default to the units of offset_start_element",
                self,
            )
            self.required_period_units = None
        if time_cell_method:
            if isinstance(time_cell_method, iris.coords.CellMethod):
                pass
            elif isinstance(time_cell_method, dict):
                method = time_cell_method.get("method")
                time_cell_method = iris.coords.CellMethod(
                    method,
                    **{k: v for k, v in time_cell_method.items() if k != "method"},
                )
            else:
                raise TypeError(
                    "time_cell_method must be either an iris.coords.CellMethod or a dict"
                )
            if "pre_save_callback" in self:

                def new_callback(cube):
                    cube.add_cell_method(cube)
                    return self["pre_save_callback"](cube)

            else:

                def new_callback(cube):
                    cube.add_cell_method(time_cell_method)
                    return cube

            self["pre_save_callback"] = new_callback

    def configure_units(self, units_dict):
        super().configure_units(units_dict)
        if not self.offset_start_units:
            self.offset_start_units = units_dict.get(self.offset_start_element, None)
        if not self.offset_end_units:
            self.offset_end_units = units_dict.get(self.offset_end_element, None)
        if not self.required_period_units:
            self.required_period_units = self.offset_start_units

    def get_time_cell(self, ob):
        point = self.element_dict["timestamp_callback"](ob)
        start = point + cf_units.Unit(self.offset_start_units).convert(
            ob[self.offset_start_element], IRIS_TIMESTAMP_UNIT
        )
        end = point + cf_units.Unit(self.offset_end_units).convert(
            ob[self.offset_end_element], IRIS_TIMESTAMP_UNIT
        )
        return (point, (start, end))

    def valid_ob(self, ob) -> bool:
        for element_name in self._metdb_element_names:
            if element_name not in ob or np.ma.is_masked(ob[element_name]):
                return False

        if self.required_period:
            _, (start, end) = self.get_time_cell(ob)
            period = cf_units.Unit(self.required_period_units).convert(
                self.required_period, IRIS_TIMESTAMP_UNIT
            )
            if not np.isclose(end - start, period):
                return False
        return True


class UnityPseudoElement(SimpleUniqueIDElementMixin, ElementBase):
    """Class with a value of 1 for any successful retrieval.

    This class is intended for implementing a counter. For example
    ATDNET Lightning strikes can be counted just by querying MetDB
    for location + time information.
    """

    def __init__(self, **kwargs) -> None:
        if "element_name" in kwargs:
            raise ValueError("UnityPseudoElement may not have any element_name(s)")
        super().__init__(**kwargs)

    def valid_ob(self, ob) -> bool:
        return True

    @property
    def metdb_elemnt_names(self):
        return set()

    def get_time_cell(self, ob):
        return (self.element_dict["timestamp_callback"](ob), None)

    def get_element_value_from_ob(self, ob):
        """Gets the value matching this element from a MetDB observation"""

        return 1


class ObsBase(abc.ABC):
    """Abstract Base Class for extracting data from MetDB"""

    # ensure class has REQUIRED_ELEMENT_KEYS set
    @classmethod
    @property
    @abc.abstractmethod
    def REQUIRED_ELEMENT_KEYS(cls):
        pass

    # ensure class has a subtype set
    @classmethod
    @property
    @abc.abstractmethod
    def _subtype(cls):
        pass

    def __init__(self, *args, contact: str = None, **kwargs) -> None:
        """Initialise ObsBase

        Args:
            contact (str): Email address to use as as MetDB contact.

        Raises:
            ValueError: Raised if contact is not provided.
            NotImplementedError: Raised if a subclass does not set _subtype
        """

        if contact is None:
            raise ValueError("contact keyword argument not supplied")
        self._contact = contact

        self._grid_cube = None
        self._latbounds = None
        self._lonbounds = None
        self._pole = None
        self.grid_cube = kwargs.get("grid_cube", None)

        self._units_lookup = (
            oemplotlib.CONFIG["metdb"]
            .get(self._subtype, {})
            .get("unit conversions", {})
        )

        self._special_type_lookup = (
            oemplotlib.CONFIG["metdb"].get(self._subtype, {}).get("special types", {})
        )

        self._common_elements = {}
        self._obs = None
        self._retrieval_elements = None

    @property
    def observations(self):
        """Returns the last set of observations retrieved from Metdb"""
        return self._obs

    @property
    def retrieval_elements(self):
        """Returns the last set of elements used for a MetDB retrieval"""
        return self._retrieval_elements

    @property
    def grid_cube(self):
        """Returns the grid_cube.

        Returns the cube defining the area over which observations will be retrieved.
        For a global area this may be None.
        """
        return self._grid_cube

    @grid_cube.setter
    def grid_cube(self, cube: iris.cube.Cube):
        """Sets the grid cube.

        Sets the grid cube as well s related private members that
        define the grid over which observations should be retrieved.

        Args:
            cube (iris.cube.Cube): Cube defining a 2D grid or None.
                                   None will result in a retrieval over
                                   the whole globe.

        Raises:
            ValueError: If the supplied cube uses an unsupported coordinate system.
                        Note that only a subset of iris.coord.coordinate_systems
                        are supported.
        """
        self._grid_cube = cube
        if cube is None:
            self._latbounds = None
            self._lonbounds = None
            self._pole = None
        else:
            lat, lon = oemplotlib.utils.get_lat_lon_from_cube(cube)
            self._latbounds = (min(lat.points), max(lat.points))

            # for metdb compatibility ensure longitude is in -180 to 180 range
            # before taking max/min
            minlon = oemplotlib.utils.fix_longitude_bounds(min(lon.points))
            maxlon = oemplotlib.utils.fix_longitude_bounds(max(lon.points))
            self._lonbounds = (min([minlon, maxlon]), max([minlon, maxlon]))

            coord_system = lat.coord_system
            if isinstance(coord_system, iris.coord_systems.RotatedGeogCS):
                pole_lat = coord_system.grid_north_pole_latitude
                pole_lon = oemplotlib.utils.fix_longitude_bounds(
                    coord_system.grid_north_pole_longitude
                )
                self._pole = (pole_lon, pole_lat)
            elif isinstance(coord_system, iris.coord_systems.GeogCS):
                # reset pole in case we've change the grid from an unrotated
                # to a rotated system
                self._pole = None
            else:
                raise ValueError("cube is in unsupported coordinate-system")

    @property
    def grid_pole(self):
        """Returns the (rotated) pole of the grid being used for extractions or None for Global"""
        if (self._pole is None) or (self._pole[0] is None) or (self._pole[1] is None):
            return None
        return self._pole

    @property
    def grid_bounds(self):
        """Returns a nested tuple defining the grid used for extractions or None for Global.

        If a tuple is returned it is
        ((min_longitude, max_longitude), (min_latitude, max_latitude))
        """
        if self._lonbounds is None or self._latbounds is None:
            return None
        return (self._lonbounds, self._latbounds)

    @staticmethod
    def add_ns(latitude: float) -> str:
        """Adds N (north) or S (south) suffixes to latitudes

        Args:
            latitude (float): Latitude to be suffixed

        Returns:
            str: Latitude with an N or S suffix,
                 note that the - symbol will be removed from southerly latitudes
        """
        if latitude >= 0:
            return f"{latitude}N"
        return f"{abs(latitude)}S"

    @staticmethod
    def add_ew(longitude: float) -> str:
        """Adds E (east) or W (west) suffixes to longitudes

        Args:
            longitude (float): Longitude to be suffixed

        Returns:
            str: Longitude with an E or W suffix,
                 note that the - symbol will be removed from westerly latitudes
        """
        if longitude >= 0:
            return f"{longitude}E"
        return f"{abs(longitude)}W"

    @staticmethod
    def _append_elements(old_elements, new_elements):
        """Combines strings tuples or lists in to a single list of elements"""

        out_elements = copy.deepcopy(old_elements)
        if isinstance(new_elements, list) or isinstance(new_elements, tuple):
            out_elements.extend(copy.deepcopy(new_elements))
        else:
            out_elements.append(copy.deepcopy(new_elements))
        return out_elements

    @abc.abstractmethod
    def get_obs(self, *args, **kwargs):
        """Fetch observations from MetDB"""

        raise NotImplementedError("get_obs must be overridden by a sub-class")

    def _call_metdb(self, keywords, elements):
        """The actual call to the MetDB interface itself"""

        LOGGER.debug(
            "Calling MetDB with:\n %s %s keywords: %s elements: %s",
            self._contact,
            self._subtype,
            keywords,
            elements,
        )
        return list(
            metdb.iter_obs(
                self._contact, self._subtype, keywords=keywords, elements=elements
            )
        )

    @staticmethod
    def _save(cube, output_folder, *args, **kwargs):
        """Save a single cube to an output folder"""

        iris.save(
            cube,
            output_folder
            / (oemplotlib.utils.filesafe_string(cube.name()) + SAVE_FILE_SUFFIX),
        )


class SiteObsABC(ObsBase, abc.ABC):
    """Base class for Site based MetDB retrievals.

    Children of this class are retrieve data from a WMO monitoring site.
    They expect the returned observations to be be capable of including
    latitude, longitude and time of observation information.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._time_elements = ["%Y", "%m", "%d", "%H", "%M"]
        self._common_elements = {
            "latitude": "LTTD",
            "longitude": "LNGD",
            "%Y": "YEAR",
            "%m": "MNTH",
            "%d": "DAY",
            "%H": "HOUR",
            "%M": "MINT",
        }

    def get_obs(self, start_time, end_time, element_dicts: list) -> list:
        """Fetch observations from MetDB

        Observations are fetched from MetDB for each element in the list
        of element_dicts. Multiple elmements are combined in to a single
        MetDB retirieval and will be included as entries in in the results
        dict.

        The elements used for the retrieval will be stored in the 'retrieval_elements'
        member of the class, and the final observations will be stored in the 'observations'
        member as well as be returned by the function.

        start_time and end_time must have a strftime method.

        Args:
            start_time (datetime like): Start of observation window
            end_time (datetime like): End of observaton window
            element_dicts (list): List of elements to be used in the retrieval

        Raises:
            ValueError: Raised if one of elements is missing required values for the retrieval
            ValueError: Raised if the 'unique_id' value for an element not in the required format

        Returns:
            list: List of returned observations, each entry is a dict as returned by metdb.iter_obs
        """

        keywords = [
            "START TIME {0:%Y%m%d/%H%MZ}".format(start_time),
            "END TIME {0:%Y%m%d/%H%MZ}".format(end_time),
        ]
        if self.grid_bounds:
            (minlon, maxlon), (minlat, maxlat) = self.grid_bounds
            keywords.append(
                f"AREA {self.add_ns(maxlat)} {self.add_ns(minlat)}"
                f" {self.add_ew(minlon)} {self.add_ew(maxlon)}",
            )
        if self.grid_pole:
            lon, lat = self.grid_pole
            keywords.append(f"RPOLE {self.add_ns(lat)} {self.add_ew(lon)}")

        metdb_elements = set(copy.copy(self._common_elements).values())
        if not isinstance(element_dicts, list):
            element_dicts = [element_dicts]
        self._retrieval_elements = []
        for element in element_dicts:
            if "timestamp_callback" in element:
                if not callable(element["timestamp_callback"]):
                    raise ValueError("timestamp_callback must be callable")
            else:
                element["timestamp_callback"] = self._ob_to_timestamp

            unique_id = element.get("unique_id", None)
            if not unique_id:
                raise ValueError(f"unique_id not found in element {element}")

            if not isinstance(element, ElementBase):
                if element.get("element_name", None) in self._special_type_lookup:
                    special_element = self._special_type_lookup[element["element_name"]]
                    ElementType = globals()[special_element["type"]]
                    element_kwargs = special_element["extra kwargs"].copy()
                    element_kwargs.update(element)
                else:
                    ElementType = SimpleElement
                    element_kwargs = element
                element = ElementType(**element_kwargs)

            # call unique id property for validation
            element.unique_id_element_names

            self._retrieval_elements.append(element)

            for key in self.REQUIRED_ELEMENT_KEYS:
                if key not in element:
                    raise ValueError(f"'{key}' missing from element {element}")

            element.configure_units(self._units_lookup)

            metdb_elements = metdb_elements | element.metdb_elemnt_names
        metdb_elements = list(metdb_elements)
        self._obs = super()._call_metdb(keywords, metdb_elements)

        return self.observations

    @staticmethod
    def _make_unique_id(request, observation):
        """Convert multiple entries of 'unique_id' in to a single entry for indexing"""
        if isinstance(request, str):
            return observation[request]
        elif callable(request):
            return request(observation)
        elif isinstance(request, tuple):
            return request[-1](observation)
        else:
            raise ValueError("Unable to make unique id")

    def save_as_cubes(
        self, output_folder, elements: list = None, observations: list = None
    ):
        """Saves observations to an output_folder as cubes in netcdf files.

        By default this converts the last set of observations retrieved in to cubes and
        saves them to netcdf file. If both a list of elements and a list of observations are
        supplied, they will be converted and saved instead.

        Args:
            output_folder (pathlib.Path): Path of a folder in which cubes should be saved
            elements (list, optional): List of element dicts that were requested. Defaults to None.
            observations (list, optional): List of observation dicts that were retrieved. Defaults to None.

        Raises:
            ValueError: Raised if only one of 'elements' or 'observations' is supplied.
        """
        if elements is None and observations is None:
            elements = self.retrieval_elements
            observations = self.observations
        elif (elements is None and observations is not None) or (
            elements is not None and observations is None
        ):
            # observations and elements are linked, so we can't just use the internal value with
            # a value handed in to the method. We require the user to be explicit.
            raise ValueError(
                "either both elements and observations must be provided, or neither"
            )
        for element, cube in self.cubes_from_obs(elements, observations):
            if "pre_save_callback" in element:
                cube = element["pre_save_callback"](cube)
            self._save(cube, output_folder=output_folder)

    def _ob_to_timestamp(self, ob):
        return datetime.datetime(
            ob[self._common_elements["%Y"]],
            ob[self._common_elements["%m"]],
            ob[self._common_elements["%d"]],
            hour=ob[self._common_elements["%H"]],
            minute=ob[self._common_elements["%M"]],
            tzinfo=datetime.timezone.utc,
        ).timestamp() / float(SECONDS_IN_HOUR)

    def cubes_from_obs(
        self,
        elements: list,
        observations: list,
        cube_postproc_callback: Callable[[dict, iris.cube.Cube], None] = None,
    ) -> iris.cube.CubeList:
        """Converts observations in to a list of cubes

        For a given element a cube of dimensions unique_id, time is created that
        covers all unique_ids and times in the results. Where observations were
        available for a given id and time this is populated with the observation,
        otherwise it is set to np.NaN.

        Args:
            elements (list): List of elements dicts used for the MetDB retrieval
            observations (list): List of observation dicts returned from MetDB
            cube_postproc_callback (Callable[[dict, iris.cube.Cube], None], optional):
                        Callback function taking an element dict and an iris Cube as arguments.
                        This is called after the cube has been constructed and may be used to
                        modify it in place. The element dict is the one used to make the request
                        to MetDB and the cube is the observations that were returned.
                        Defaults to None.

        Returns:
            list : List containing tuples of (element dict, cube of observations).
                   Element is the element that was used in the MetDB request to make
                   the cube of observations. Note that this list may be shorter than the
                   original list of elements as any that did not result in observations
                   being retrieved are skipped.
        """

        final_elments_cubes = []

        for element in elements:
            name = element["common_name"]
            LOGGER.info("Converting %s observations to cube", name)

            # We need to build a Nd cube covering the whole dataset
            # i.e. all site ids and times and then populate it with obs.
            # We can't just rely on iris merge/concatenate as not all sites
            # site report observations for all times so merge/concatenate
            # results in multiple cubes. Our cube is initialised with np.NaN
            # to fill in gaps where observations were not reported.

            # work out all possible coordinate values

            # first coordinates defined by the element
            extra_coords = element.get_extra_static_coords()

            # now coordinates that depend on the observations retrieved
            all_ob_times = set()
            all_ob_sites = {}

            for ob in observations:
                if not element.valid_ob(ob):
                    continue
                all_ob_times.add(element.get_time_cell(ob))
                unique_id = self._make_unique_id(element["unique_id"], ob)
                lon_lat = (
                    ob[self._common_elements["longitude"]],
                    ob[self._common_elements["latitude"]],
                )
                if unique_id in all_ob_sites and all_ob_sites[unique_id] != lon_lat:
                    LOGGER.warning(
                        "Site %s already in list for element %s, replacing (lon, lat) %s with %s",
                        unique_id,
                        element,
                        all_ob_sites[unique_id],
                        lon_lat,
                    )
                all_ob_sites[unique_id] = lon_lat

            # now build master cube to hold observations

            # sort obs by id
            all_ob_ids = sorted(all_ob_sites.keys())

            if len(all_ob_ids) == 0:
                LOGGER.warning("No sites found for element %s, skipping", element)
                continue
            if len(all_ob_times) == 0:
                LOGGER.warning("No times found for element %s, skipping", element)
                continue

            # make coords
            id_coord = iris.coords.DimCoord(np.array(all_ob_ids))
            id_coord.rename("unique_id")
            lon_coord = iris.coords.AuxCoord(
                np.array([all_ob_sites[s][0] for s in all_ob_ids]),
                standard_name="longitude",
                coord_system=iris.coord_systems.GeogCS(
                    iris.fileformats.pp.EARTH_RADIUS
                ),
                units="degrees",
            )
            lat_coord = iris.coords.AuxCoord(
                np.array([all_ob_sites[s][1] for s in all_ob_ids]),
                standard_name="latitude",
                coord_system=iris.coord_systems.GeogCS(
                    iris.fileformats.pp.EARTH_RADIUS
                ),
                units="degrees",
            )
            all_ob_times = sorted(list(all_ob_times), key=lambda s: s[0])

            time_points = [p[0] for p in all_ob_times]
            time_bounds = [p[1] if len(p) == 2 else None for p in all_ob_times]
            if any(time_bounds):
                if not all(time_bounds):
                    raise ValueError(
                        f"Observation {element} has a mixture of bounded and unbounded times"
                    )
                time_bounds = np.array(time_bounds)
            else:
                time_bounds = None

            time_coord = iris.coords.DimCoord(
                np.array(time_points),
                bounds=time_bounds,
                standard_name="time",
                units="hours since 1970-01-01 00:00:00",
            )

            dim_coords = [id_coord, time_coord]
            dim_coord_names = [c.name() for c in dim_coords]
            aux_coords = [(lat_coord, 0), (lon_coord, 0)]

            for extra_dim, extra_aux_coords in extra_coords:
                if isinstance(extra_dim, str):
                    try:
                        dim_index = dim_coord_names.index(extra_dim)
                    except ValueError as err:
                        raise ValueError(
                            f"expected pre-existing {extra_dim} coordinate not found"
                        ) from err
                else:
                    extra_name = extra_dim.name()
                    if extra_name in dim_coord_names:
                        raise ValueError("Unable to make extra coord, already exists")
                    dim_coords.append(extra_dim)
                    dim_coord_names.append(extra_name)
                    dim_index = len(dim_coords)
                for aux in extra_aux_coords:
                    aux_coords.append((aux, dim_index))

            cube_shape = tuple(len(c.points) for c in dim_coords)

            # Finally, make an empty data array and use it to build a cube
            output_dtype = np.float32
            output_nan_marker = np.NaN
            data = np.empty(cube_shape, output_dtype)
            data[:] = output_nan_marker
            cube = iris.cube.Cube(
                data,
                dim_coords_and_dims=[(c, i) for i, c in enumerate(dim_coords)],
                aux_coords_and_dims=aux_coords,
            )
            if name:
                cube.rename(name)
            if "units" in element:
                cube.units = element["units"]

            # now populate cube with observations
            for ob in observations:
                obvalue = np.ma.filled(
                    np.asarray(element.get_element_value_from_ob(ob), output_dtype),
                    output_nan_marker,
                )
                if not element.valid_ob(ob):
                    continue
                ob_id = self._make_unique_id(element["unique_id"], ob)
                obtime = element.get_time_cell(ob)[0]
                ob_id_index = np.where(cube.coord("unique_id").points == ob_id)
                time_index = np.where(cube.coord("time").points == obtime)
                cube.data[ob_id_index, time_index] = obvalue

            if cube_postproc_callback:
                cube_postproc_callback(element, cube)
            final_elments_cubes.append((element, cube))
        if len(final_elments_cubes) == 0:
            raise ValueError("No cubes made from observations")
        return final_elments_cubes


class LNDSYNObs(SiteObsABC):
    """Class for retrieving MetDB LNDSYN subtypes"""

    REQUIRED_ELEMENT_KEYS = ["common_name", "element_name"]
    _subtype = "LNDSYN"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_obs(self, start_time, end_time, element_dicts):
        """Ensures that unique_id is specified for all elements before observation retrieval"""
        element_dicts = copy.deepcopy(element_dicts)
        if not isinstance(element_dicts, list):
            element_dicts = [element_dicts]
        for element_dict in element_dicts:
            if "unique_id" not in element_dict:
                element_dict["unique_id"] = (
                    "WMO_STTN_NMBR",
                    "WMO_BLCK_NMBR",
                    self.make_unique_id,
                )
        return super().get_obs(start_time, end_time, element_dicts)

    @staticmethod
    def make_unique_id(ob_entry):
        """Combines WMO block number and WMO station number in to a single number"""
        return 1000 * ob_entry["WMO_BLCK_NMBR"] + ob_entry["WMO_STTN_NMBR"]


class LNDSYBObs(LNDSYNObs):
    """Class for retrieving MetDB LNDSYB subytpes"""

    _subtype = "LNDSYB"


class SREWObs(SiteObsABC):
    """Class for retrieving MetDB SREW subtypes"""

    REQUIRED_ELEMENT_KEYS = ["common_name", "element_name"]
    _subtype = "SREW"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # SREWs don't return minutes so remove from common elements
        self._common_elements.pop("%M", None)

    # SREWs don't return minutes so override timestamp conversion
    def _ob_to_timestamp(self, ob):
        return datetime.datetime(
            ob[self._common_elements["%Y"]],
            ob[self._common_elements["%m"]],
            ob[self._common_elements["%d"]],
            hour=ob[self._common_elements["%H"]],
            tzinfo=datetime.timezone.utc,
        ).timestamp() / float(SECONDS_IN_HOUR)

    def get_obs(self, start_time, end_time, element_dicts):
        """Ensures that unique_id is specified for all elements before observation retrieval"""

        element_dicts = copy.deepcopy(element_dicts)
        if not isinstance(element_dicts, list):
            element_dicts = [element_dicts]
        for element_dict in element_dicts:
            if "unique_id" not in element_dict:
                element_dict["unique_id"] = "WMO_STTN_INDX_NMBR"
        return super().get_obs(start_time, end_time, element_dicts)


class ATDNETObs(SiteObsABC):

    REQUIRED_ELEMENT_KEYS = ["common_name"]
    _subtype = "ATDNET"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._time_elements.append("%S")
        self._common_elements["%S"] = "SCND"
        self._unique_id_register = {}

    def _ob_to_timestamp(self, ob):
        return datetime.datetime(
            ob[self._common_elements["%Y"]],
            ob[self._common_elements["%m"]],
            ob[self._common_elements["%d"]],
            hour=ob[self._common_elements["%H"]],
            minute=ob[self._common_elements["%M"]],
            second=ob[self._common_elements["%S"]],
            tzinfo=datetime.timezone.utc,
        ).timestamp() / float(SECONDS_IN_HOUR)

    def _unique_id_callback(self, observation):
        identifier = tuple(
            observation[c] for c in sorted(self._common_elements.values())
        )
        unique_id = self._unique_id_register.get(identifier, None)
        if unique_id:
            return unique_id
        else:
            new_id = len(self._unique_id_register) + 1
            self._unique_id_register[identifier] = new_id
            return new_id

    def get_obs(self, start_time, end_time, element_dicts):
        """Ensures that unique_id is specified for all elements before observation retrieval"""

        element_dicts = copy.deepcopy(element_dicts)
        if not isinstance(element_dicts, list):
            element_dicts = [element_dicts]
        for element_dict in element_dicts:
            if "unique_id" not in element_dict:
                element_dict["unique_id"] = self._unique_id_callback
        return super().get_obs(start_time, end_time, element_dicts)
