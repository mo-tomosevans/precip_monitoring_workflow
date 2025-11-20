import logging
import configobj
from pathlib import Path

LOGGER = logging.getLogger(__name__)

LOGGER.info("loadingconfig %s", str(Path(__file__).parent.resolve() / "config.conf"))

CONFIG = configobj.ConfigObj(
    str(Path(__file__).parent.resolve() / "config.conf"), unrepr=True
)

DEFAULT_TIME_FMT = "%Y%m%dT%H%M"


class _FcstPeriodSmartFmt:
    def __init__(self) -> None:
        # Forecast period: 3 decimal places is enough to show individual minutes
        # when the value to be formatted is in hours, or quarter hours when the
        # value is in days
        self._ndp = 3
        self._num_fmt = "{:." + str(self._ndp) + "f}"
        self._positive_format = "T+{:." + str(self._ndp) + "f}"
        self._negative_format = "T-{:." + str(self._ndp) + "f}"

    def __str__(self) -> str:
        return self._positive_format

    def format(self, value):
        test_fmt = self._num_fmt.format(value)
        if test_fmt == "-" + self._num_fmt.format(0.0):
            return self._positive_format.format(0.0)
        elif value < 0:
            return self._negative_format.format(-1.0 * value)
        return self._positive_format.format(value)


FCST_PERIOD_FMT = _FcstPeriodSmartFmt()

PLOT_FILE_EXTENSIONS = ["png"]
DEFAULT_PLOT_EXTENSION = "png"

DEFAULT_CUBE_TIME_COORD = "time"
DEFAULT_CUBE_FC_PERIOD_COORD = "forecast_period"
DEFAULT_CUBE_FC_REF_COORD = "forecast_reference_time"
DEFAULT_CUBE_MEMBER_COORD = "realization"

METDB_UNIQUE_ID_COORD = "unique_id"

DEFAULT_PLOT_TAR_NAME = "plots.tar"

NONESTR = "None"

REPLICATION_DIM_NAME = "replication"

from . import plots
from . import colorbars
from . import loaders
from . import utils
from . import gridtools
from . import cube_utils
from . import stats

try:
    import moxie as _moxie
except ImportError:
    LOGGER.warning("Unable to import moxie, oemplotlib.moxieplots will be unavailable")
else:
    from . import moxieplots

try:
    import metdb as _metdb
except ImportError:
    LOGGER.warning("Unable to import metdb, oemplotlib.metdb will be unavailable.")
else:
    from . import metdb
