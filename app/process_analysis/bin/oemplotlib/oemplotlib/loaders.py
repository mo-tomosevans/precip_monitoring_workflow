import logging
import re
import abc
import glob
import numpy as np
import iris
from . import nimrod_to_cubes as n2c
from . import CONFIG

LOGGER = logging.getLogger(__name__)


def _load_once(cubeorlist):
    if cubeorlist not in ["cubelist", "cube"]:
        raise ValueError("_load_once decorator requires a cube or cubelist argument")

    def load_once_decorator(fn):
        def load_once(self, *args, **kwargs):
            target = getattr(self, cubeorlist)
            if target is None:
                return fn(self, *args, **kwargs)
            return target

        return load_once

    return load_once_decorator


class CubeLoaderABC(abc.ABC):
    def __init__(self, files, constraint=None):
        if isinstance(files, list):
            self._orrig_files = [str(f) for f in files]
        else:
            self._orrig_files = str(files)

        self._orrig_constraint = constraint
        self.cube = None
        self.cubelist = None

    @abc.abstractmethod
    def load(self):
        pass

    @abc.abstractmethod
    def load_cube(self):
        pass


class DetPPLoader(CubeLoaderABC):
    @_load_once("cubelist")
    def load(self):
        self.cubelist = iris.load(self._orrig_files, self._orrig_constraint)
        return self.cubelist

    @_load_once("cube")
    def load_cube(self):
        self.cube = iris.load_cube(self._orrig_files, self._orrig_constraint)
        return self.cube


class EnsPPLoader(CubeLoaderABC):
    def __init__(self, files, constraint=None):
        super().__init__(files, constraint)
        self._regexes = [
            re.compile(reg) for reg in CONFIG["data loading"]["realization regexes"]
        ]

    @_load_once("cubelist")
    def load(self):
        self.cubelist = iris.load(
            self._orrig_files, self._orrig_constraint, callback=self._load_callback
        )
        return self.cubelist

    @_load_once("cube")
    def load_cube(self):
        self.cube = iris.load_cube(
            self._orrig_files, self._orrig_constraint, callback=self._load_callback
        )
        return self.cube

    def _load_callback(self, cube, field, filename):
        if not cube.coords("realization"):
            LOGGER.warning(
                "%s cube from %s is missing realization", cube.name(), filename
            )
            for regex in self._regexes:
                match = regex.match(filename)
                LOGGER.debug("checking %s for match... %s", filename, match)
                if match:
                    ensemble_coord = iris.coords.AuxCoord(
                        np.int32(match["member"]),
                        standard_name="realization",
                        units="1",
                    )
                    cube.add_aux_coord(ensemble_coord)

                    LOGGER.debug(
                        "%s, added realization %s from filename %s",
                        cube.name(),
                        match["member"],
                        filename,
                    )
                    break
            else:
                raise ValueError("Unable to determine realization from file name")


class PPLoader(CubeLoaderABC):
    """A loader class that automatically determines whether it is loading ensemble data"""

    def __init__(self, files, constraint=None):
        super().__init__(files, constraint)
        self._ens_regexes = [
            re.compile(reg) for reg in CONFIG["data loading"]["realization regexes"]
        ]
        self._loader = (EnsPPLoader if self._loading_ensembles() else DetPPLoader)(
            files, constraint=constraint
        )

    def load(self):
        self.cubes = self._loader.load()
        return self.cubes

    def load_cube(self):
        self.cubes = self._loader.load_cube()
        return self.cubes

    def _loading_ensembles(self) -> bool:
        """Test if the loader is loading ensembles

        Raises:
            ValueError: Raised if a mixture of ensemble and deterministic
                        files are found.

        Returns:
            (bool): True if loading ensembles, otherwise False
        """

        filenames = []
        [filenames.extend(glob.glob(f)) for f in self._orrig_files]
        LOGGER.debug("_loading_ensembles checking filenames \n%s", filenames)
        matches = []
        for filename in filenames:
            for reg in self._ens_regexes:
                if reg.match(filename):
                    matches.append(True)
                    break
            else:
                matches.append(False)

        if all(matches):
            LOGGER.info("Loading ensemble pp files")
            return True
        elif any(matches):
            raise ValueError("Mixture of ensemble and non-ensemble files")
        LOGGER.info("Loading deterministic pp files")
        return False


class RadarAccumLoader(CubeLoaderABC):
    def __init__(self, data_files, weight_files, constraint=None):
        super().__init__(data_files, constraint)
        self._weight_files = weight_files
        self._raw_weight_cubes = None

    @_load_once("cubelist")
    def load(self):
        self._load_raw()
        self._make_masked_cubes()
        return self.cubelist

    @_load_once("cube")
    def load_cube(self):
        self._load_raw()
        self._make_masked_cubes()
        if len(self.cubelist) > 1:
            raise ValueError("More than one radar cube loaded")
        self.cube = self.cubelist[0]
        return self.cube

    def _load_raw(self):
        self._raw_data_cubes = n2c.nimrod_to_cubes(
            self._orrig_files, constraint=self._orrig_constraint
        )
        self._raw_weight_cubes = n2c.nimrod_to_cubes(
            self._weight_files, constraint=self._orrig_constraint
        )

    def _make_masked_cubes(self):
        weights = self._raw_weight_cubes.copy()
        self.cubelist = iris.cube.CubeList()
        for i, w in enumerate(weights):
            w.data *= 32
            w.units = "1"
            mask = w.data < 11
            data_for_mask = np.ma.array(self._raw_data_cubes[i].data, mask=mask)
            self.cubelist.append(self._raw_data_cubes[i].copy(data=data_for_mask))
