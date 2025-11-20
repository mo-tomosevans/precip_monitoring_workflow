from abc import ABC, abstractmethod
import numpy as np
import iris
from scipy import ndimage, signal
from cf_units import Unit

from oemplotlib import LOGGER as _OEMLOGGER
from oemplotlib.utils import ens_safe_slicer
import oemplotlib.utils

LOGGER = _OEMLOGGER.getChild(__name__)

try:
    from moxie.multiprocessing import parallel

    _PARALLEL_STATS = True
except ImportError:
    LOGGER.error("oemplotlib.stats: moxie not available, using serial processing")
    _PARALLEL_STATS = False

# Currently we are implementing our own neighbourhood methods
# but in the future we may want to update to use IMPROVER code
# so we will mimic their interface in the hope that this makes
# it easy to upgrade in the future.
# HOWEVER, IMPROVER has requirements for the coordinates and
# metadata for its cubes that we will not attempt to
# mimic as this will add a lot of complexity.


class ImproverBasePlugin(ABC):
    """An abstract class that mimics the interface for IMPROVER plugins ABC.
    Subclasses must be callable. We preserve the process
    method by redirecting to __call__.
    """

    def __call__(self, *args, **kwargs):
        """Makes subclasses callable to use process
        Args:
            *args:
                Positional arguments.
            **kwargs:
                Keyword arguments.
        Returns:
            Output of self.process()
        """
        return self.process(*args, **kwargs)

    @abstractmethod
    def process(self, *args, **kwargs):
        """Abstract class for rest to implement."""
        pass


class OEMStatsABC(ImproverBasePlugin):
    """Base ABC for all OEM neighbourhood processing classes"""

    def __init__(self, stat_name=None, *args, **kwargs) -> None:
        if stat_name is None:
            raise NotImplementedError("stat_name must be specified by a child class")

        self._stat_name_base = stat_name

    def __call__(self, cube: iris.cube.Cube, *args, **kwargs) -> iris.cube.Cube:
        return super().__call__(cube, *args, **kwargs)

    @abstractmethod
    def process(self, cube: iris.cube.Cube, *args, **kwargs) -> iris.cube.Cube:
        return cube

    def get_names(
        self,
        cube,
        param_name=None,
        layer_name_template=None,
        full_name_template=None,
        stat_name=None,
        meteo_category=None,
        test_str=None,
        period_name_str=None,
        use_full_layer_name=False,
        *args,
        **kwargs,
    ):
        """Return names describing the modified cube.

        Returns a 'layer name' describing the cube in human readable format
        and a 'file name' suitable for saving plots produced from the cube.

        .. note::
           This will only work out appropriate names for the cube returned by the process
           method. It will not perform any calculations or modify the input cube.

        .. note::
           By default the 'layer' name returned will not include
           'period' or 'test' components

        Args:
            cube (iris.cube.Cube): The iris cube for which the names should be calculated
            layer_name_template: A template to use to work out a short name for the layer.
                                 This should be a python string that can be formatted with
                                 placeholders for 'param_name' and 'stat_name'.
            full_name_template (str): A template to use to work out the names.
                                      This should be a python string that can be formatted
                                      with placeholders for 'param_name' and 'stat_name'.
                                      If 'period_name_str' and/or 'test_str' is set the template
                                      must also have a placeholder for it/them.
            param_name (str, optional): The name of the parameter for which the statistic
                                        is being calculated. Defaults to cube.name().
            stat_name (str, optional): The name of the statistic being calculated.
                                       Defaults to the name set when initialising the class.
            meteo_category (str, optional): The meteorological category under which this
                                            layer should be grouped (e.g. Temperature, Snow)
            test_str (str, optional): A string describing any 'test' that has been
                                      applied to the cube
            period_name_str (str, optional): The name of the time period over which
                                             any time processing has been applied to
                                             the cube
            full_layer_name (Boolean, optional): If true use the full_name_template
                                                 to calculate the layer_name,
                                                 otherwise use the layer_name_template.
                                                 Defaults to False.

        Returns:
            tuple[str,str]: A tuple containing the 'layer name' and 'file name' strings
        """

        assert full_name_template is not None, "full_name_template must be specified"
        if test_str and "test_str" not in full_name_template:
            raise ValueError("test_str is set but not included in name_template")
        if period_name_str and "period_name_str" not in full_name_template:
            raise ValueError("period_name_str is set but not included in name_template")
        if not use_full_layer_name:
            assert (
                layer_name_template is not None
            ), "layer_name_template must be specified when not using full layer name"

        param_name = param_name if param_name else cube.name()
        stat_name = stat_name if stat_name else self._stat_name_base
        meteo_category = "" if meteo_category is None else meteo_category

        test_str = test_str if test_str else ""
        period_name_str = period_name_str if period_name_str else ""

        full_name = full_name_template.format(
            param_name=param_name,
            stat_name=stat_name,
            test_str=test_str,
            period_name_str=period_name_str,
        )
        file_name = oemplotlib.utils.replace_prob_strings(full_name)
        if meteo_category:
            file_name = f"{meteo_category} - {file_name}"

        if use_full_layer_name:
            layer_name = full_name
        else:
            layer_name = layer_name_template.format(
                param_name=param_name,
                stat_name=stat_name,
                test_str=test_str,
                period_name_str=period_name_str,
            )
        return layer_name, file_name

    def _memberprocess(
        self,
        cube,
        memberfn,
        param_name=None,
        member_units=Unit("unknown"),
    ):
        """A process that should be applied to each member of the input cube

        Args:
            cube (iris.cube.Cube): A cube containing one or more ensemble members.
                                   Deterministic cubes will be handled as if they contain
                                   a single member.
            memberfn (callable): A callable that should take a 2 dimensional cube
                                 as an input and return a 2 dimensional cube.
            param_name (str, optional): The name of the parameter for which the statistic
                                        is being calculated. Defaults to cube.name().
            member_units (cf_units.Unit, optional): The unit to which the cube output by
                                                    memberfn should be set.
                                                    Defaults to Unit("unknown").
                                                    .. note::
                                                       The unit will be set, not converted.
                                                       No consistency checking is performed.


        Returns:
            iris.cube.Cube: A cube created by calling memberfn on each member of input cube.
        """
        assert callable(memberfn), "memberfn must be callable"

        full_name, _ = self.get_names(cube, param_name)

        tmp_cubelist = iris.cube.CubeList()
        lat_coord, lon_coord = oemplotlib.utils.get_lat_lon_from_cube(cube)

        if _PARALLEL_STATS:

            @parallel
            def memberfn_wrapper(mem_cube):
                return memberfn(mem_cube)

        else:

            def memberfn_wrapper(mem_cube):
                latlon_cubelist = iris.cube.CubeList()
                for latlong_slice in mem_cube.slices(
                    [lat_coord, lon_coord], ordered=False
                ):

                    latlon_result_cube = memberfn(latlong_slice)

                    latlon_cubelist.append(latlon_result_cube)
                return latlon_cubelist.merge_cube()

        for mem_cube, _ in ens_safe_slicer(cube):

            latlon_result_cube = memberfn_wrapper(mem_cube)
            latlon_result_cube.rename(full_name)
            latlon_result_cube.units = member_units
            tmp_cubelist.append(latlon_result_cube)

        result_cube = tmp_cubelist.merge_cube()

        return result_cube

    def _allmembersprocess(
        self,
        cube,
        memberfn,
        param_name=None,
        member_units=Unit("unknown"),
    ):
        """A process that can be applied to all members of the cube at once

        Args:
            cube (iris.cube.Cube): A cube containing one or more ensemble members.
                                   Deterministic cubes will be handled as if they contain
                                   a single member.
            memberfn (callable): A callable that should take cube as its input and return a cube.
            param_name (str, optional): The name of the parameter for which the statistic
                                        is being calculated. Defaults to cube.name().
            member_units (cf_units.Unit, optional): The unit to which the cube output by
                                                    memberfn should be set.
                                                    Defaults to Unit("unknown").
                                                    .. note::
                                                       The unit will be set, not converted.
                                                       No consistency checking is performed.


        Returns:
            iris.cube.Cube: A cube created by calling memberfn on each member of input cube.
        """
        assert callable(memberfn), "memberfn must be callable"

        full_name, _ = self.get_names(cube, param_name)

        lat_coord, lon_coord = oemplotlib.utils.get_lat_lon_from_cube(cube)
        try:
            cube.coord(oemplotlib.DEFAULT_CUBE_MEMBER_COORD)
            slice_cords = [oemplotlib.DEFAULT_CUBE_MEMBER_COORD, lat_coord, lon_coord]
        except iris.exceptions.CoordinateNotFoundError:
            slice_cords = [lat_coord, lon_coord]

        if _PARALLEL_STATS:

            @parallel
            def memberfn_wrapper(mem_cube):
                return memberfn(mem_cube)

        else:

            def memberfn_wrapper(mem_cube):
                latlon_cubelist = iris.cube.CubeList()
                for latlong_slice in mem_cube.slices(slice_cords, ordered=False):
                    latlon_result_cube = memberfn(latlong_slice)
                    latlon_cubelist.append(latlon_result_cube)
                return latlon_cubelist.merge_cube()

        result_cube = memberfn_wrapper(cube)
        result_cube.rename(full_name)

        result_cube.units = member_units

        return result_cube


class BinaryProbs(OEMStatsABC):
    """Converts a cube of model data to a cube of binary probabilities"""

    def __init__(
        self, test=None, stat_name="Prob", test_str=None, *args, **kwargs
    ) -> None:
        if test is None:
            raise ValueError("test must be specified")
        self.test = test
        self.test_str = test_str

        super().__init__(stat_name=stat_name, *args, **kwargs)

    def get_names(
        self,
        cube,
        param_name=None,
        stat_name=None,
        meteo_category=None,
        period_name_str=None,
        use_full_layer_name=False,
    ):
        test_str = f" ({self.test_str})" if self.test_str else ""
        period_name_template = f" ({{period_name_str:s}})" if period_name_str else ""
        layer_name_template = f"{{param_name:s}} {{stat_name:s}}"
        full_name_template = (
            f"{{param_name:s}}{period_name_template} {{stat_name:s}}{{test_str:s}}"
        )

        return super().get_names(
            cube,
            param_name=param_name,
            layer_name_template=layer_name_template,
            full_name_template=full_name_template,
            stat_name=stat_name,
            meteo_category=meteo_category,
            test_str=test_str,
            period_name_str=period_name_str,
            use_full_layer_name=use_full_layer_name,
        )

    def process(self, cube, param_name=None, *args, **kwargs):

        memfn = lambda mem_cube: mem_cube.copy(
            data=np.ma.where(self.test(mem_cube), 1, 0)
        )

        return self._allmembersprocess(
            cube, memfn, param_name=param_name, member_units="1"
        )


class _NeibhbourhoodMixin:
    def __init__(self, neighbourhood_size=1, *args, **kwargs) -> None:
        self._neighbourhood_size = None
        self.neighbourhood_size = neighbourhood_size
        super().__init__(*args, **kwargs)

    def get_names(
        self,
        cube,
        param_name=None,
        stat_name=None,
        meteo_category=None,
        period_name_str=None,
        use_full_layer_name=False,
    ):

        period_name_template = f" ({{period_name_str:s}})" if period_name_str else ""
        layer_name_template = (
            f"{{param_name:s}} {{stat_name:s}}(Size{self.neighbourhood_size})"
        )
        full_name_template = f"{{param_name:s}}{period_name_template} {{stat_name:s}}{{test_str:s}}(Size{self.neighbourhood_size})"

        return super().get_names(
            cube,
            param_name=param_name,
            layer_name_template=layer_name_template,
            full_name_template=full_name_template,
            stat_name=stat_name,
            meteo_category=meteo_category,
            period_name_str=period_name_str,
            use_full_layer_name=use_full_layer_name,
        )

    @property
    def neighbourhood_size(self):
        """The size of the neighbourhood over which operations should be calculated

        .. note ::
           Setting this to a new value will override the value used when
           initializing the class and change the size of the neighbourhood used
           for all future calculations.
        """
        return self._neighbourhood_size

    @neighbourhood_size.setter
    def neighbourhood_size(self, size):
        self._neighbourhood_size = size


class NeighbourhoodMax(_NeibhbourhoodMixin, OEMStatsABC):
    """Calculates the maximum value in a 2d square neighbourhood"""

    def __init__(self, stat_name="Neighbourhood Max", *args, **kwargs) -> None:

        super().__init__(stat_name=stat_name, *args, **kwargs)

    def process(self, cube, param_name=None, *args, **kwargs) -> iris.cube.Cube:

        result = super().process(cube, *args, **kwargs)

        kernel = np.ones((self.neighbourhood_size, self.neighbourhood_size))

        memberfn = lambda mem_cube: mem_cube.copy(
            data=ndimage.maximum_filter(mem_cube.data, footprint=kernel, mode="nearest")
        )

        return self._memberprocess(
            result, memberfn, param_name=param_name, member_units=result.units
        )


class NeighbourhoodMeanConvolve(_NeibhbourhoodMixin, OEMStatsABC):
    """Calculate a neighbourhood 'mean' using a 2d square convolution"""

    def __init__(self, stat_name="Neighbourhood Mean", *args, **kwargs) -> None:

        super().__init__(stat_name=stat_name, *args, **kwargs)

    def process(self, cube, param_name=None, *args, **kwargs) -> iris.cube.Cube:

        result = super().process(cube, *args, **kwargs)

        kernel = np.ones((self.neighbourhood_size, self.neighbourhood_size))

        memberfn = lambda mem_cube: mem_cube.copy(
            data=(
                signal.convolve2d(mem_cube.data, kernel, boundary="symm", mode="same")
                / kernel.sum()
            )
        )

        return self._memberprocess(
            result, memberfn, param_name=param_name, member_units=result.units
        )
