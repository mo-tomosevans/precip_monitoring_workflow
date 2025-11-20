import copy
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.colors as mcol
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import oemplotlib
from oemplotlib.utils import expand_range_string, expand_range_string_list
from oemplotlib import CONFIG

LOGGER = oemplotlib.LOGGER.getChild(__name__)

try:
    from moxie.plot import colors as moxcolors
except ImportError:
    LOGGER.warning(
        "Unable to import moxie, moxie derived colorbars will be unavailable"
    )
    moxcolors = None


class Colorbar:
    def __init__(
        self,
        cmap=None,
        extend=None,
        bounds=None,
        unit=None,
        ticks=None,
        values=None,
        *args,
        **kwargs
    ):
        self.__cmap = cmap
        self.__extend = extend
        self.__bounds = bounds
        self.__values = values
        self.__unit = unit
        self.__extra_kwargs = None
        if ticks:
            self.__ticks = ticks
        elif ticks is None and bounds:
            self.__ticks = bounds
        else:
            self.__ticks = None
        self.configuration = kwargs

    @property
    def cmap(self):
        return self.__cmap

    @property
    def extend(self):
        return self.__extend

    @property
    def bounds(self):
        return self.__bounds

    @property
    def ticks(self):
        return self.__ticks

    @property
    def unit(self):
        return self.__unit

    @property
    def values(self):
        return self.__values

    @property
    def norm(self):
        if self.__bounds:
            return mcol.BoundaryNorm(self.__bounds, self.__cmap.N)
        else:
            return mcol.Normalize()

    @property
    def configuration(self):
        return self.__extra_kwargs

    @configuration.setter
    def configuration(self, kwargs):
        self.__extra_kwargs = copy.deepcopy(kwargs)

    def make_colorbar(self, mappable, *args, **kwargs):
        cbarkwargs = copy.deepcopy(self.configuration)
        if self.__ticks:
            cbarkwargs.update({"ticks": self.__ticks})
        if self.__values:
            cbarkwargs.update({"values": self.__values})
        cbarkwargs.update(kwargs)
        ax = plt.gca()
        if cbarkwargs.get("orientation", None) != "horizontal":
            f = plt.gcf()
            divider = make_axes_locatable(ax)
            ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
            f.add_axes(ax_cb)
            cbarkwargs["cax"] = ax_cb
        LOGGER.debug("making colorbar with kwargs %s", cbarkwargs)
        cbar = plt.colorbar(mappable, *args, **cbarkwargs)
        cbar.set_label(self.__unit)
        # make sure the sublot axis is the current axis
        # so future modifications aren't applied to the
        # colorbar Axis
        plt.sca(ax)
        return cbar

    @classmethod
    def from_mpl_colormap(cls, cmap_name, *args, **kwargs):
        return cls(matplotlib.cm.get_cmap(cmap_name), *args, **kwargs)

    def make_transparent(self):
        cmap_colors = self.cmap(np.arange(self.cmap.N))
        cmap_colors[:, -1] = np.linspace(0, 1, self.cmap.N)

        self.__cmap = mcol.ListedColormap(cmap_colors)


class NullCbar(Colorbar):
    @property
    def norm(self):
        return None

    def make_colorbar(self, mappable, *args, **kwargs):
        return None


class LabelSize10Mixin:
    def make_colorbar(self, mappable, *args, **kwargs):
        cbar = super().make_colorbar(mappable, *args, **kwargs)
        cbar.ax.tick_params(labelsize=10)
        return cbar


def _load_cmap(colormap_input, range=None):
    cmap_path = Path(__file__).parent / "colormaps" / colormap_input
    LOGGER.debug("Looking for colormap file %s", cmap_path)
    if cmap_path.exists():
        LOGGER.debug("found colormap file, loading")
        rgb = np.loadtxt(cmap_path)
        cmap = mcol.ListedColormap(rgb, N=rgb.shape[0])
    else:
        LOGGER.debug(
            "file not found, looking for matplotlib %s colormap", colormap_input
        )
        cmap = matplotlib.cm.get_cmap(colormap_input)
    if range:
        rgb = cmap(list(range))
        cmap = mcol.ListedColormap(rgb, N=rgb.shape[0])
    return cmap


def _make_config_class(cls_name):
    settings = CONFIG["colorbars"][cls_name]

    LOGGER.debug(
        "Making colorbar %s with settings %s", cls_name, CONFIG["colorbars"][cls_name]
    )

    parent_classes = [globals()[cls] for cls in settings.get("inherit", ["Colorbar"])]
    if parent_classes[-1] is not Colorbar:
        parent_classes += [Colorbar]

    LOGGER.debug("setting up colormap")
    colormap_input = settings["colormap"]
    colormap_range = settings.get("colormap_subrange", None)

    if isinstance(colormap_range, str):
        colormap_range = expand_range_string(colormap_range)
    if isinstance(colormap_input, str):
        if colormap_input.lstrip().startswith("moxie."):
            if moxcolors is None:
                LOGGER.warning("Moxie not available, skipping %s", cls_name)
                raise ValueError("Moxie not available")
            default_cmap = moxcolors.get_cmap(colormap_input.lstrip()[len("moxie.") :])
        else:
            default_cmap = _load_cmap(colormap_input, colormap_range)
    else:
        LOGGER.debug("Creating colormap from input variable(s)")
        default_cmap = mcol.ListedColormap(colormap_input)
    LOGGER.debug("made default colormap %s %s", type(default_cmap), default_cmap)

    default_bounds = settings.get("bounds", None)
    if default_bounds:
        default_bounds = expand_range_string_list(default_bounds)

    default_ticks = settings.get("ticks", None)
    if default_ticks:
        default_ticks = expand_range_string_list(default_ticks)

    default_values = settings.get("values", None)
    if default_values:
        default_values = expand_range_string_list(default_values)

    default_unit = settings.get("unit", None)

    default_extend = settings.get("extend", None)

    class TmpCbar(*parent_classes):
        def __init__(
            self,
            cmap=None,
            extend=None,
            bounds=None,
            unit=None,
            ticks=False,
            values=None,
            *args,
            **kwargs
        ):
            super().__init__(
                cmap=cmap if cmap else default_cmap,
                extend=extend if extend else default_extend,
                bounds=bounds if bounds else default_bounds,
                unit=unit if unit else default_unit,
                ticks=ticks if ticks else default_ticks,
                values=values if values else default_values,
                *args,
                **kwargs
            )

    return TmpCbar


_n_loaded_cbars = 0
for cbar in CONFIG["colorbars"]:
    try:
        globals()[cbar] = _make_config_class(cbar)
        _n_loaded_cbars += 1
    except ValueError as exp:
        LOGGER.warning("Unable to load colorbar %s", cbar)
if _n_loaded_cbars == 0:
    raise ImportError("Unable to create any colorbars")
