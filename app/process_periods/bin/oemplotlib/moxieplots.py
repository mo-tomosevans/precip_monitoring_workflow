from __future__ import annotations
import os
import socket
import pathlib
from typing import List, Union, Type
from queue import Empty as QEmptyError
from contextlib import contextmanager
import numpy as np
import iris
import moxie
import multiprocessing
import matplotlib.pyplot as plt
import ImageMetaTag as imt
import oemplotlib
from oemplotlib.moxiesavehandlers import SaveHandlerABC, NullHandler

LOGGER = oemplotlib.LOGGER.getChild(__name__)

CYLC_CREATION_CYCLE = os.environ.get("CYLC_TASK_CYCLE_POINT", None)
CYLC_CREATION_CYCLE_KEY = "CYLC_CREATION_CYCLE"

_MOXIE_ID_Q = multiprocessing.Queue()
_MOXIE_ID_INITIAL_PROC = multiprocessing.current_process().pid
_MOXIE_ID_INITIAL_HOST = "".join(c for c in socket.gethostname() if c.isalnum())
# moxie.plot.colors._continuous_levels defines a set number
# of levels for making "continuous" colorbars
# (i.e. plots with discrete=False)
_MOXIE_N_CONTINUOUS_COLORS = 256

# the correct way to get the number of available processors on a batch system is
# len(os.sched_getaffinity(0)), however moxie is using cpu_count.
# We override the setting here to avoid oversubscribing when running on batch systems.
moxie.config._config["num_cpus"] = max(
    1, min(moxie.config.get("num_cpus"), len(os.sched_getaffinity(0)))
)

# Use the matplotlib default font as this supports more unicode symbols
# than moxie's default.
moxie.config._config["font"] = "DejaVu Sans"

# We will read the value from the moxie config in order to avoid triggering the fallback
#  mechanisms for labelling temporary imt databases
for i in range(1, moxie.config.get("num_cpus") + 1):
    _MOXIE_ID_Q.put(i)


class _OEMLayerLayout(moxie.plot.layout.Layout):
    @property
    def colorbar_ticklabels(self):
        """
        Returns
        -------
        extent : float
            Extent of space required for the chart's colorbar ticks, in inches.
            For an OEM Layer this must be a constant size and not
            vary with the ticklabel contents.

        """
        if self.chart._colorbar is None:
            return 0
        return (
            self.pad
            * oemplotlib.CONFIG["layout"]["colorbar"]["ticklabel_padding_factor"]
        )


class _DecorateMixin:
    "A mixin class that automatically adds plot details"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._oem_colorbar_orientation = kwargs.pop(
            "colorbar_orientation",
            oemplotlib.CONFIG["layout"]["colorbar"].get("default_orientation", None),
        )
        self._oem_decorated = False

    def save(self, fpath=None, **kwargs):
        self.decorate()
        return super().save(fpath=fpath, **kwargs)

    def decorate(self, force=False):
        "Automatically add map elements before saving"
        if self._oem_decorated and not force:
            LOGGER.debug(
                "Decorate called more than once, ignoring. "
                "Use force=True to override this."
            )
            # decorate should usually only be called once
            return
        if self._colorbar is None:
            if self._oem_colorbar_orientation:
                # we can't pass in orientation=None here as moxie
                # will pass any value set through to matplotlib
                # thereby ignoring the moxie default
                self.colorbar(orientation=self._oem_colorbar_orientation)
            else:
                self.colorbar()
        if not any(
            getattr(node, "plot_type", None) == "coastlines" for node in self._nodelist
        ):
            self.coastlines()

        # Moxie batch plotting will defer setting the title and save it in _deferred_calls
        title_already_set = any(
            call.method == "title" for call in getattr(self, "_deferred_calls", [])
        )
        if not self._title and not title_already_set:
            self.title()
        self._oem_decorated = True


class _AutoCbarMixin:
    "A mixin class that applies an OEM Colorbar to a moxie Chart"

    def __init__(self, **kwargs):
        self._oem_cbar = kwargs.pop("oem_colorbar", None)
        self.oem_cbar_invert = kwargs.pop("oem_colorbar_invert", None)
        super().__init__(**kwargs)

    def _make_oem_kwargs(
        self,
        attrs: list[str | tuple] = None,
        cbar: oemplotlib.colorbars.Colorbar = None,
        plot_kwargs: dict | None = None,
    ) -> dict:
        """Extracts oem keyword settings from the pre-set colorbar.

        Settings are extracted for the pre-set colorbar without any validation.
        This function simply maps attributes in the colorbar to keys in a dictionary.

        Args:
            attrs (List[str | tuple], optional): List of attributes to be extracted
                from the colorbar.
                If the attribute is a string it must be the name of an attribute
                of the colorbar. If the attribute is a 2 tuple the first value should
                be the key to be added to the oem keyword dict and the second
                value should be the name of the attribute of the colorbar to look up.
                Defaults to None.
            cbar (optional): The colorbar to extract the attributes from. If not provided
                the colorbar used to initialise the class instance will be used.
            plot_kwargs: Additional keyword arguments to be included in the returned dict.
                By default these will override any attributes extracted from the colorbar.
                However, if the arguments include `discrete=False`, this will be set to `True`
                if the colormap for the chosen colorbar does not have enough colours
                to support moxie's 'continuous' mode.

        Returns:
            dict: A dictionary of keyword arguments that could be massed to a
                  moxie Chart plotting function.
        """
        # default to empty dict
        oem_kwargs = {}

        override_kwargs = plot_kwargs.copy()

        if (self._oem_cbar is None and cbar is None) or attrs is None:
            # nothing to do
            oem_kwargs.update(override_kwargs)
            return oem_kwargs

        cbar = self._oem_cbar if cbar is None else cbar

        if plot_kwargs.get("discrete", None) is False:
            # if discrete is set to True, but the colormap does not have enough colours
            # to support moxie's discrete mode, set it to False

            Ncolors = cbar.cmap.N
            if Ncolors < _MOXIE_N_CONTINUOUS_COLORS - 2:
                LOGGER.warning(
                    "Non-discrete mode requested but colorbar (%s) "
                    "does not have enough colours (%s/%s), setting discrete=True "
                    "and manually setting levels to use the full colormap",
                    cbar.__class__.__name__,
                    Ncolors,
                    _MOXIE_N_CONTINUOUS_COLORS,
                )
                override_kwargs["discrete"] = True

                # recreate moxie.plot.colors._continuous_levels
                # but use the number of levels in the colormap instead of
                # _MOXIE_N_CONTINUOUS_COLORS

                def _continuous_levels(levels):
                    extended_levels = np.linspace(
                        levels[0], levels[-1], Ncolors - len(levels) + 2
                    )

                    # Make sure every level from the original list is in the extended list
                    levels = np.append(extended_levels, levels[1:-1])
                    return sorted(levels)

                override_kwargs["levels"] = _continuous_levels(cbar.bounds)

        for attr in attrs:
            if isinstance(attr, tuple):
                kwarg_key, cbar_attr_name = attr
            else:
                kwarg_key = cbar_attr_name = attr
            cbar_attr = getattr(cbar, cbar_attr_name, None)
            if cbar_attr:
                oem_kwargs[kwarg_key] = cbar_attr

        oem_kwargs.update(override_kwargs)
        return oem_kwargs

    def contour(self, cube, layer="default", **kwargs):
        """Set up default kwargs before calling parent's contour method"""

        oem_kwargs = self._make_oem_kwargs(
            cbar=kwargs.pop("cbar", None), plot_kwargs=kwargs
        )

        return super().contour(cube, layer=layer, **oem_kwargs)

    def pcolormesh(self, cube, layer="default", **kwargs):
        """Set up default kwargs before calling parent's pcolormesh method"""

        oem_kwargs = self._make_oem_kwargs(
            ["cmap", ("levels", "bounds")],
            cbar=kwargs.pop("cbar", None),
            plot_kwargs=kwargs,
        )

        return super().pcolormesh(cube, layer=layer, **oem_kwargs)

    def scatter(self, cube, layer="default", **kwargs):
        """Set up default kwargs before calling parent's scatter method"""

        oem_kwargs = self._make_oem_kwargs(
            ["cmap", ("levels", "bounds")],
            cbar=kwargs.pop("cbar", None),
            plot_kwargs=kwargs,
        )

        return super().scatter(cube, layer=layer, **oem_kwargs)

    def colorbar(self, layer=None, **kwargs):
        """Set up default kwargs before calling parent's colorbar method"""

        cbar = kwargs["cbar"] if "cbar" in kwargs else self._oem_cbar

        oem_kwargs = self._make_oem_kwargs(
            [
                ("boundaries", "bounds"),
                ("label", "moxie_unit_label"),
                "values",
                "extend",
            ],
            cbar=cbar,
            plot_kwargs=kwargs,
        )

        ticks = cbar.ticks

        # Newer versions of moxie+matplotlib require manually specified
        # ticks not to include the value that gets automatically removed
        # when the colorbar is extended.

        if oem_kwargs.get("extend", None) == "min":
            tick_range = slice(1)
        elif oem_kwargs.get("extend", None) == "max":
            tick_range = slice(0, -1)
        elif oem_kwargs.get("extend", None) == "both":
            tick_range = slice(1, -1)
        elif oem_kwargs.get("extend", None) in ["neither", None]:
            tick_range = slice(None)
        else:
            raise ValueError(
                f"_AutoCbarMixin: Unknown colorbar extend value {oem_kwargs.get('extend', None)}"
            )

        if ticks:
            ticks = ticks[tick_range]

        tmp_var = super().colorbar(
            layer=layer,
            ticks=ticks,
            **oem_kwargs,
        )

        if self.oem_cbar_invert:
            self._colorbar.ax.invert_yaxis()

        return tmp_var


class _GridHandlerMixin:
    "A mixin class that handles setting up the domain"

    def __init__(self, **kwargs) -> None:
        handler = kwargs.pop("grid_handler", None)
        if handler:
            gridkwargs = {
                "crs": handler.plotting_projection,
                "domain": handler.plotting_domain,
            }
        else:
            gridkwargs = {}

        gridkwargs.update(kwargs)

        super().__init__(**gridkwargs)


class _IMTMixin:
    "A mixin class that adds IMT database capabilities"

    def __init__(self, **kwargs):

        self.imt_dir = kwargs.pop("imt_dir")
        self.imt_prefix = "{}".format(
            kwargs.pop("imt_prefix", oemplotlib.CONFIG["imt"]["prefix"])
        )
        self.imt_extra_tags = kwargs.pop("imt_extra_tags", {})
        self.imt_multiprocessing_token = kwargs.pop("imt_multiprocessing_token", None)

        super().__init__(**kwargs)

    @staticmethod
    def _oem_safepath(fpath):
        def make_safe_output(fpath):
            # make sure file name doesn't contain spaces, illegal characters, etc
            # we ignore the directory here as imt saving won't make it, so
            # it should already be safe if it exists.
            fpath = pathlib.Path(fpath)
            fdir = fpath.parent
            fname = oemplotlib.utils.filesafe_string(str(fpath.name))

            return str(fdir / fname)

        fpath = str(fpath)

        for ext in oemplotlib.PLOT_FILE_EXTENSIONS:
            if fpath.endswith(f".{ext}"):
                # strip extension then make safe
                return (make_safe_output(".".join(fpath.split(".")[:-1])), ext)

        parts = fpath.split(".")
        return (
            f"{make_safe_output('.'.join(parts[:-1]))}.{parts[-1]}",
            oemplotlib.DEFAULT_PLOT_EXTENSION,
        )

    def save(self, fpath=None, **kwargs):
        fpath_out = super().save(fpath=fpath, **kwargs)

        # the _save_image method changes the file name
        # but that doesn't get passed back through moxie
        # so apply the same file name change here
        fpath_out, ext_out = self._oem_safepath(fpath_out)
        return f"{fpath_out}.{ext_out}"

    def _save_image(self, fpath):
        imt_fname = "{}_{}.db".format(
            self.imt_prefix,
            (
                self.imt_multiprocessing_token
                if self.imt_multiprocessing_token
                else f"{_MOXIE_ID_INITIAL_HOST}_{multiprocessing.current_process().pid}"
            ),
        )

        cube = None
        for node in moxie.plot.charts.nodes.NodeList(self[self._get_base_layer()]):
            if node.cube is None:
                continue
            if isinstance(node.cube, iris.cube.CubeList):
                cube = node.cube[0]
            else:
                cube = node.cube
        if cube is None:
            raise ValueError("Unable to find cube to determine IMT tags")

        labeler = oemplotlib.plots.PlotLabeler(cube)

        img_tags = labeler.imt_tags(imt_tags=self.imt_extra_tags, force_member=True)

        img_tags[CYLC_CREATION_CYCLE_KEY] = CYLC_CREATION_CYCLE

        fname, fextension = self._oem_safepath(fpath)

        imt.savefig(
            fname,
            fig=self.figure,
            img_tags=img_tags,
            db_file=str(os.path.join(self.imt_dir, imt_fname)),
            img_format=fextension,
            keep_open=True,
        )


class _SaveHandlerMixin:
    """A mixin that overrides the saving process"""

    _DEFAULT_OEM_SAVE_HANDLER = NullHandler()

    def __init__(self, **kwargs) -> None:
        self._oem_save_handler = kwargs.pop("save_handler", None)

        if self._oem_save_handler is None:
            # NOTE: if the default behaviour is changed from
            #       'do nothing' the Batch class will also
            #        need to be updated
            self._oem_save_handler = self._DEFAULT_OEM_SAVE_HANDLER
        if not isinstance(self._oem_save_handler, SaveHandlerABC):
            raise TypeError(
                "save_handler kwarg must be an instance of moxiesavehandlers.SaveHandlerABC"
            )
        super().__init__(**kwargs)

    @classmethod
    @contextmanager
    def using_save_handler_context(cls, handler: SaveHandlerABC):
        """Open a context to override the default SaveHandler

        Opens a context within which the handler provided will be used
        unless an override is provided when the an instance of the class
        is initialised.

        .. note::
           The handler context will be opened as long as this
           context is open.

        Args:
            handler (SaveHandlerABC): An instance of a save handler to use
                                      in place of the default.

        Yields:
            Iterator[Type[_SaveHandlerMixin]]: The class on which this method was called
        """
        try:
            previous_handler = cls._DEFAULT_OEM_SAVE_HANDLER
            cls._DEFAULT_OEM_SAVE_HANDLER = handler
            with handler:
                yield cls
        finally:
            cls._DEFAULT_OEM_SAVE_HANDLER = previous_handler

    def save(self, fpath=None, **kwargs):

        # use the save handler to rewrite the required file path
        # before calling save, then use the save handler to manipulate
        # the output files
        working_fpath = self._oem_save_handler.get_working_fname(fpath)
        return self._oem_save_handler.handle_output_files(
            super().save(fpath=working_fpath, **kwargs)
        )


class _OEMLayerBase(moxie.plot.charts.Overlay):
    @staticmethod
    def _get_oem_style_kwargs(method_name):
        return (
            oemplotlib.CONFIG.get("style", {})
            .get("plot kwargs", {})
            .get(method_name, {})
            .copy()
        )

    def contour(self, *args, **kwargs):
        style_kwargs = self._get_oem_style_kwargs("contour")
        style_kwargs.update(kwargs)
        return super().contour(*args, **style_kwargs)

    def pcolormesh(self, *args, **kwargs):
        style_kwargs = self._get_oem_style_kwargs("pcolormesh")
        style_kwargs.update(kwargs)
        return super().pcolormesh(*args, **style_kwargs)

    def scatter(self, *args, **kwargs):
        style_kwargs = self._get_oem_style_kwargs("scatter")
        style_kwargs.update(kwargs)
        return super().scatter(*args, **style_kwargs)

    def quiver(self, *args, **kwargs):
        style_kwargs = self._get_oem_style_kwargs("quiver")
        style_kwargs.update(kwargs)
        return super().quiver(*args, **style_kwargs)

    def colorbar(self, *args, **kwargs):

        cbar = super().colorbar(*args, **kwargs)

        # We are using a slightly newer version of matplotlib
        # where the colorbar axis patch needs to be made transparent
        # as well as the colorbar patch.
        # TODO - When moxie is updated to expect a different environment
        #        check if this workaround is still needed.
        self._colorbar.ax.patch.set_alpha(0)

        return cbar

    def save(self, fpath=None, **kwargs):

        # moxie bug: moxie uses fpath for its Chart
        # class and Batch class, but then fname in the method for
        # its Overlay class, however this never causes problems as
        # it is always passed to super().save as a positional argument.
        # We can avoid triggering the bug by doing the same here.
        return super().save(fpath, **kwargs)


class _VisibleColorbarMixin:
    """A Mixin Class that ensures the colorbar is visible"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._colorbar_background = None

    def colorbar(self, *args, **kwargs):
        """
        Add a colorbar to the chart in order to achieve a layout
        which matches a base image that has a colorbar.

        """

        # OEMLayer requires a visible colorbar so call the moxie Chart version
        moxie.plot.charts.Chart.colorbar(self, *args, **kwargs)

        if self._colorbar_background is None:
            # Add background to make sure the topmost colorbar covers any other layers

            bbox = self._colorbar.ax.get_tightbbox(self.figure.canvas.get_renderer())

            self._colorbar_background = plt.Rectangle(
                (bbox.xmin, bbox.ymin),
                bbox.width,
                bbox.height,
                fill=True,
                color="white",
                alpha=1.0,
                zorder=-1,
                # bbox in pixel coordinates so no transform required
                transform=None,
                figure=self.figure,
            )
            self.figure.patches.extend([self._colorbar_background])

    def fix_layout(self):
        """
        Apply an automatically generated tidy layout to the chart.

        """

        super().fix_layout()
        if self._colorbar_background:
            bbox = self._colorbar.ax.get_tightbbox(self.figure.canvas.get_renderer())
            self._colorbar_background.set_bounds(
                bbox.xmin, bbox.ymin, bbox.width, bbox.height
            )


class _VisibleTitleMixin:
    """A Mixin Class that ensures the title is visible"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._title_background = None

    def title(self, *args, **kwargs):
        """Add a visible title to the plot"""

        kwargs["alpha"] = 1
        # We want the title to be visible so call the Chart base class
        moxie.plot.charts.Chart.title(self, *args, **kwargs)

        # Add background so only the topmost title shows.
        if self.layout.title > 0 and self._title_background is None:
            for ax_x0, ax_y0, ax_w, ax_h in self.layout.get_axes_pos():
                # get_axes_pos is a generator but only returns a single value
                self._title_background = plt.Rectangle(
                    (ax_x0, ax_y0 + ax_h),
                    1,  # 1 = full width
                    self.layout.title,
                    fill=True,
                    color="white",
                    alpha=1.0,
                    zorder=-1,
                    # transform moxie figure coordinates to pixel coordinates
                    transform=self.figure.transFigure,
                    figure=self.figure,
                )
                self.figure.patches.extend([self._title_background])

    def fix_layout(self):
        """
        Apply an automatically generated tidy layout to the chart.

        """

        super().fix_layout()
        if self._title_background and self.layout.title > 0:
            for ax_x0, ax_y0, ax_w, ax_h in self.layout.get_axes_pos():
                self._title_background.set_bounds(
                    ax_x0, ax_y0 + ax_h, 1, self.layout.title  # 1 = full width
                )


class OEMLayer(
    _SaveHandlerMixin,
    _AutoCbarMixin,
    _DecorateMixin,
    _IMTMixin,
    _VisibleTitleMixin,
    _VisibleColorbarMixin,
    _GridHandlerMixin,
    _OEMLayerBase,
):
    "A transparent layer for making interactive OEM plots"

    def __init__(self, **kwargs):

        layout_kwargs = kwargs.get("layout_kwargs", {})
        super().__init__(**kwargs)
        # replace default layout
        self.layout = _OEMLayerLayout(self, **layout_kwargs)


class OEMContourLayer(
    _SaveHandlerMixin,
    _AutoCbarMixin,
    _DecorateMixin,
    _IMTMixin,
    _VisibleTitleMixin,
    _GridHandlerMixin,
    _OEMLayerBase,
):
    "A transparent contour layer for making interactive OEM plots"

    def __init__(self, **kwargs):

        layout_kwargs = kwargs.get("layout_kwargs", {})
        super().__init__(**kwargs)
        # replace default layout
        self.layout = _OEMLayerLayout(self, **layout_kwargs)


class OEMScatterLayer(
    _SaveHandlerMixin,
    _AutoCbarMixin,
    _DecorateMixin,
    _IMTMixin,
    _VisibleTitleMixin,
    _GridHandlerMixin,
    _OEMLayerBase,
):
    """A transparent scatter layer for making interactive OEM plots

    This class provides a workaround required for making scatter plot layers
    with no colorbar. This is only required if the first cube plotted on the
    layer is a 1D scatter cube, in which case it the setup method of the
    underlying moxie class would fail.
    """

    def __init__(self, **kwargs):

        layout_kwargs = kwargs.get("layout_kwargs", {})
        super().__init__(**kwargs)
        # replace default layout
        self.layout = _OEMLayerLayout(self, **layout_kwargs)

    def setup(self, cube):
        """
        Plot a transparent scatter background in order to prepare the
        Overlay for additional layers (e.g. coastlines).
        Parameters
        ----------
        cube : iris.cube.Cube
            The cube to plot (invisibly).
        """
        return self.scatter(cube, alpha=0, layer=0)


class Batch(_SaveHandlerMixin, moxie.plot.batch.Batch):

    # NOTE: Inheriting from _SaveHandlerMixin here means that the 'save_handler'
    #       kwarg is removed from the kwargs when the Batch class' __init__
    #       method is called. This means the individual plotters fall back to the
    #       default handling (i.e. do nothing) in the parallel processes. The
    #       Batch class can then manipulate the files in its save method after
    #       they have all been created.

    DEFERRED_METHODS = moxie.plot.batch.Batch.DEFERRED_METHODS + ["decorate"]

    def save(self, fpath=None, **kwargs):
        self.decorate()
        return super().save(fpath=fpath, **kwargs)

    def _execute(self, in_queue, out_queue, plot_calls, extra_calls, kwargs):
        try:
            # wait for at most 10 seconds to get a token
            imt_multiprocessing_token = _MOXIE_ID_Q.get(True, 10)
            LOGGER.debug("Got moxieplots process token %s", imt_multiprocessing_token)
        except QEmptyError:
            # fall back to default behaviour
            LOGGER.debug("Unable to get moxieplots process token")
            imt_multiprocessing_token = None
        try:
            if issubclass(self._parent, _IMTMixin) and imt_multiprocessing_token:
                self._initkwargsbatch[
                    "imt_multiprocessing_token"
                ] = f"{_MOXIE_ID_INITIAL_HOST}_{_MOXIE_ID_INITIAL_PROC}_{imt_multiprocessing_token}"
            super()._execute(in_queue, out_queue, plot_calls, extra_calls, kwargs)
        finally:
            if imt_multiprocessing_token:
                LOGGER.debug(
                    "Releasing moxieplots process token %s", imt_multiprocessing_token
                )
                _MOXIE_ID_Q.put(imt_multiprocessing_token)
