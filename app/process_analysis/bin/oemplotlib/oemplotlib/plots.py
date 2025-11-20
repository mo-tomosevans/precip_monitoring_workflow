import abc
import copy
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import patch
import iris
import iris.plot as iplt
import cartopy.crs as ccrs
import ImageMetaTag as imt

import oemplotlib
from oemplotlib import LOGGER

DEFAULT_MEMBER_STRING = "N/A"


class PlotLabeler:
    # TODO FOR ENSEMBLES THIS WILL NEED REALIZATION
    def __init__(
        self,
        cube,
        name=None,
        fcst_time=None,
        fcst_ref_time=None,
        fcst_period=None,
        fcst_time_fmt=None,
        fcst_ref_time_fmt=None,
        fcst_period_fmt=oemplotlib.FCST_PERIOD_FMT,
        realization=None,
    ):
        self._name = cube.name() if name is None else name

        self._fcst_time = (
            cube.coord("time").cell(0).point if fcst_time is None else fcst_time
        )

        if fcst_ref_time is None:
            try:
                self._fcst_ref_time = (
                    cube.coord("forecast_reference_time").cell(0).point
                )
            except iris.exceptions.CoordinateNotFoundError:
                # observations do not nave a forecast reference time
                self._fcst_ref_time = None
        else:
            self._fcst_ref_time = fcst_ref_time

        if fcst_period is None:
            try:
                self._fcst_period = fcst_period_fmt.format(
                    cube.coord("forecast_period").cell(0).point
                )
                self._fcst_period_units = str(cube.coord("forecast_period").units)
            except iris.exceptions.CoordinateNotFoundError:
                # observations do not have a forecat period
                self._fcst_period = None
                self._fcst_period_units = None

        else:
            self._fcst_period = fcst_period_fmt.format(fcst_period)
            self._fcst_period_units = ""

        self._fcst_time_fmt = (
            fcst_time_fmt if fcst_time_fmt else oemplotlib.DEFAULT_TIME_FMT
        )

        self._fcst_ref_time_fmt = (
            fcst_ref_time_fmt if fcst_ref_time_fmt else oemplotlib.DEFAULT_TIME_FMT
        )

        if realization:
            self.realization = realization
        else:
            try:
                self.realization = str(cube.coord("realization").points[0])
            except iris.exceptions.CoordinateNotFoundError:
                self.realization = None

    def title(self):
        title = self._name
        if self.realization:
            title += " member {}".format(self.realization)
        if any([self._fcst_time, self._fcst_ref_time, self._fcst_period]):
            title += "\n"
        if self._fcst_time:
            title += self._fcst_time.strftime(self._fcst_time_fmt)
        if self._fcst_ref_time and self._fcst_period:
            ref_string = ": {} + {} {}".format(
                self._fcst_ref_time.strftime(self._fcst_ref_time_fmt),
                self._fcst_period,
                self._fcst_period_units,
            )
            title += ref_string
        return title

    def filename(self, suffix=None):
        fname = [self._name]
        if self._fcst_ref_time:
            fname.append(self._fcst_ref_time.strftime(self._fcst_ref_time_fmt))
        if self._fcst_time:
            fname.append(self._fcst_time.strftime(self._fcst_time_fmt))
        if self.realization:
            fname.append(self.realization)
        fname = " ".join(fname)
        fname = oemplotlib.utils.filesafe_string(fname)
        return fname

    def imt_tags(self, imt_tags: dict = None, force_member: bool = False) -> dict:
        """Generate dictionary of imt metadata

        Args:
            imt_tags (dict, optional): A dictionary of custom tags to be included.
                                       Automatically generated tags will be replaced
                                       if there is a clash. Defaults to None.
            force_member (bool, optional): If True, always include a 'member' tag,
                                           even for deterministic models.
                                           Defaults to False.

        Returns:
            dict: Dictionary of {tag_name: tag_value} pairs, containing at least
                  "plot_type", "data_day", "data_time", "valid_time" and "lead_time"
                  tags. "member" will be added for ensembles.
        """

        tags = {
            "plot_type": "_".join(self._name.split()),
            "data_day": self._fcst_ref_time.strftime("%Y%m%d")
            if self._fcst_ref_time
            else None,
            "data_time": self._fcst_ref_time.strftime("%H%M")
            if self._fcst_ref_time
            else None,
            "valid_time": self._fcst_time.strftime("%Y%m%dT%H%M"),
            "lead_time": "_".join(self._fcst_period.split())
            if self._fcst_period
            else None,
        }

        if self.realization:
            tags["member"] = self.realization
        elif force_member:
            tags["member"] = DEFAULT_MEMBER_STRING
        if imt_tags:
            tags.update(imt_tags)
        return tags


class Figure:
    def __init__(
        self,
        output_dir=None,
        labeller=None,
        title=None,
        filename=None,
        subplots=(1, 1),
        subplot_kw=None,
        use_gridspec=None,
        figsize=12,
        tight_layout=True,
        subplots_adjust=True,
        projection=ccrs.PlateCarree(),
    ):

        self.output_dir = Path(output_dir) if output_dir else None

        if not isinstance(labeller, PlotLabeler):
            raise ValueError("labeller must be an instance of PlotLabeller")
        if labeller and title:
            LOGGER.warning(
                "labeller and name specified, " "automatic name will be overriden"
            )
        if labeller and title:
            LOGGER.warning(
                "labeller and filename specified, "
                "automatic filename will be overriden"
            )
        self.set_labeller(labeller)
        if title:
            self.title = title
        if filename:
            self.filename = filename

        if isinstance(projection, iris.cube.Cube):
            self.projection = oemplotlib.utils.projection_from_cube(projection)
        elif isinstance(projection, ccrs.Projection):
            self.projection = projection
        else:
            raise ValueError(
                "projection must be either an iris Cube " "or a cartopy Projection"
            )

        self._Nsubplots = subplots
        if len(subplots) != 2:
            raise ValueError("subplots must be an iteralbe of length 2")
        _ratio = float(subplots[0]) / float(subplots[1])
        # ratio = sp_x / sp_y

        if figsize:
            if isinstance(figsize, int) or isinstance(figsize, float):
                figsize = (figsize, None)
            elif len(figsize) != 2:
                raise ValueError("Unable to convert figsize to tuple of length 2")
            if figsize[0] is None:
                self._figsize = ((figsize[1] / _ratio), (figsize[1]))
            elif figsize[1] is None:
                self._figsize = ((figsize[0]), (figsize[0] * _ratio))
            else:
                self._figsize = figsize
            LOGGER.debug("Figure using size %s", self._figsize)
        else:
            self._figsize = None

        if tight_layout is True:
            self._tight_layout = {}
        elif tight_layout:
            self._tight_layout = tight_layout
        else:
            self._tight_layout = False
        if subplots_adjust is True:
            self._subplots_adjust = {"left": 0.0125, "right": 0.96}
        elif subplots_adjust:
            self._subplots_adjust = subplots_adjust
        else:
            self._subplots_adjust = False

        self._gridspec = None
        self._gridspec_input = use_gridspec if use_gridspec else {}
        self._subplot_kw = subplot_kw if subplot_kw else {}

        self._fig = None
        self._subplots = None

    def set_labeller(self, labeller):
        self.labeller = (labeller,)
        if labeller:
            self.title = labeller.title()
            self.filename = labeller.filename()

    @property
    def fig(self):
        return self._fig

    @property
    def subplots(self):
        return self._subplots

    @property
    def gridspec(self):
        return self._gridspec

    def __enter__(self):
        if self._gridspec_input:
            gs = matplotlib.gridspec.GridSpec(
                *self._Nsubplots,
                **({} if self._gridspec_input is True else self._gridspec_input),
            )
            self._gridspec = gs
            self._fig = plt.figure()
            self._subplots = None
        else:
            subplot_kw = copy.deepcopy(self._subplot_kw)
            subsubplot_kw = subplot_kw.pop("subplot_kw", {})
            subsubplot_kw.update({"projection": self.projection})
            self._fig, self._subplots = plt.subplots(
                *self._Nsubplots,
                figsize=self._figsize,
                subplot_kw=subsubplot_kw,
                **subplot_kw,
            )
        if self._tight_layout:
            LOGGER.debug("using tight layout with kwargs: %s", self._tight_layout)
            self.fig.tight_layout(**self._tight_layout)
        if self._subplots_adjust:
            self.fig.subplots_adjust(**self._subplots_adjust)
        if self.title:
            self.fig.suptitle(self.title)

        return self

    def __exit__(self, type, value, traceback):
        plt.close(self.fig)

    def remove_extra_subplots(self, n_to_keep=None):
        if not n_to_keep:
            raise ValueError("number of subplots to keep not specified")
        if self.subplots is None:
            raise ValueError(
                "No internal record of subplots, they must be manged manually"
            )
        for ax in self.subplots.ravel()[n_to_keep:]:
            self.fig.delaxes(ax)

    def savefig(self, fname=None, **kwargs):
        if self.output_dir:
            fname = fname if fname else self.filename
            if any(
                [fname.endswith(f".{ext}") for ext in oemplotlib.PLOT_FILE_EXTENSIONS]
            ):
                self.fig.savefig(self.output_dir / fname, **kwargs)
            else:
                self.fig.savefig(
                    self.output_dir / f"{fname}.{oemplotlib.DEFAULT_PLOT_EXTENSION}",
                    **kwargs,
                )
        else:
            raise ValueError("output_dir has not been set")


class IMTFigure(Figure):
    def __init__(
        self,
        output_dir=None,
        labeller=None,
        title=None,
        filename=None,
        subplots=(1, 1),
        subplot_kw=None,
        use_gridspec=None,
        figsize=10,
        tight_layout=True,
        subplots_adjust=True,
        projection=ccrs.PlateCarree(),
        db_file=None,
    ):
        if db_file is None:
            raise ValueError("db_file must be specified")
        self.db_file = db_file

        super().__init__(
            output_dir=output_dir,
            labeller=labeller,
            title=title,
            filename=filename,
            subplots=subplots,
            subplot_kw=subplot_kw,
            use_gridspec=use_gridspec,
            figsize=figsize,
            tight_layout=tight_layout,
            subplots_adjust=subplots_adjust,
            projection=projection,
        )

    # def __enter__(self):
    #     super().__enter__(self)
    #     if self._subplots:
    #         for sp in self._subplots:
    #             sp.patch.set_alpha(0.0)
    #     return self

    def savefig(self, fname=None, img_tags=None, **kwargs):

        if not (img_tags):
            raise ValueError("img_tags must be specified")
        if self.output_dir:
            fname = fname if fname else self.filename
            if any(
                [fname.endswith(f".{ext}") for ext in oemplotlib.PLOT_FILE_EXTENSIONS]
            ):
                imt.savefig(
                    str(self.output_dir / fname),
                    img_tags=img_tags,
                    db_file=str(self.db_file),
                    **kwargs,
                )
            else:
                imt.savefig(
                    str(self.output_dir / fname),
                    img_tags=img_tags,
                    db_file=str(self.db_file),
                    img_format=oemplotlib.DEFAULT_PLOT_EXTENSION,
                    **kwargs,
                )
        else:
            raise ValueError("output_dir has not been set")


class IMTLayerFigure(IMTFigure):
    def savefig(self, fname=None, img_tags=None, **kwargs):

        if self._subplots:
            if isinstance(self._subplots, matplotlib.axes.Axes):
                toiter = [self._subplots]
            elif isinstance(self._subplots, np.ndarray):
                toiter = self._subplots.flat
            else:
                toiter = self._subplots
            for sp in toiter:
                sp.background_patch.set_alpha(0.0)

        nativesave = plt.savefig

        def savewrapper(*argssw, **kwargssw):
            nativesave(*argssw, **kwargssw, transparent=True)

        with patch("matplotlib.pyplot.savefig", savewrapper):
            return super().savefig(fname=fname, img_tags=img_tags, **kwargs)


class Plot(abc.ABC):
    @abc.abstractmethod
    def __init__(self, title=None, plot_callback=None, *args, **kwargs):
        self.title = title
        self._plot_handle = None
        self.plot_callback = plot_callback
        self._sub_kwargs = set()

    @abc.abstractmethod
    def _plot(self):
        pass

    def _finish_plot(self, *args, **kwargs):
        if self.title:
            plt.title(self.title)

    def plot(self, *args, **kwargs):
        self._plot(*args, **kwargs)
        if self.plot_callback:
            self.plot_callback()

        self._finish_plot(*args, **kwargs)


class MapPlot(Plot):
    def _finish_plot(self, *args, **kwargs):
        super()._finish_plot(*args, **kwargs)

        if isinstance(args[0], iris.cube.Cube):
            self.set_extents_from_cube(args[0])

    @staticmethod
    def set_extents_from_cube(cube):
        ax = plt.gca()
        lat, lon = oemplotlib.utils.get_lat_lon_from_cube(cube)
        ax.set_extent(
            (min(lon.points), max(lon.points), min(lat.points), max(lat.points)),
            crs=lat.coord_system.as_cartopy_crs(),
        )


class ColorbarMixin:
    def __init__(self, colorbar=None, *args, **kwargs):
        if colorbar is None:
            raise ValueError(f"colorbar must be specified for {type(self)}")
        self.__colorbar = colorbar
        super().__init__(*args, **kwargs)
        self._sub_kwargs.add("colorbar")

    @property
    def colorbar(self):
        return self.__colorbar

    def _finish_plot(self, *args, **kwargs):

        if (
            isinstance(self._plot_handle, matplotlib.contour.ContourSet)
            and len(self._plot_handle.levels) == 0
        ):
            LOGGER.warn(
                "Empty contour set found, using dummy mappable " "to make colorbar."
            )
            mappable = matplotlib.cm.ScalarMappable(
                norm=self.colorbar.norm, cmap=self.colorbar.cmap
            )
        else:
            mappable = self._plot_handle

        if kwargs.get("colorbar", True) or kwargs.get("colorbar") == {}:
            colorbar_opts = {}
            colorbar_opts.update(**kwargs.pop("colorbar", {}))
            self.colorbar.make_colorbar(mappable, **colorbar_opts)
        super()._finish_plot(*args, **kwargs)


class CoastlinesMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sub_kwargs.add("coastline")

    def _finish_plot(self, *args, **kwargs):
        if kwargs.get("coastline", True) or kwargs.get("coastline") == {}:
            coast_opts = {"resolution": "10m"}
            coast_opts.update(**kwargs.pop("coastline", {}))
            plt.gca().coastlines(**coast_opts)
        super()._finish_plot(*args, **kwargs)


class MapContourPlot(CoastlinesMixin, ColorbarMixin, MapPlot):
    def __init__(self, colorbar=None, title=None, *args, **kwargs):
        super().__init__(colorbar=colorbar, title=title, *args, **kwargs)

    def _plot(self, data, *args, **kwargs):

        contourkwargs = {"norm": self.colorbar.norm, "cmap": self.colorbar.cmap}
        contourkwargs.update(
            {k: v for k, v in kwargs.items() if k not in self._sub_kwargs}
        )

        self._plot_handle = iplt.contour(data, self.colorbar.bounds, **contourkwargs)


class MapPColormeshPlot(CoastlinesMixin, ColorbarMixin, MapPlot):
    def __init__(self, colorbar=None, title=None, *args, **kwargs):
        super().__init__(colorbar=colorbar, title=title, *args, **kwargs)

    def _plot(self, data, *args, **kwargs):

        contourkwargs = {
            "norm": self.colorbar.norm,
            "cmap": self.colorbar.cmap,
            "vmin": self.colorbar.bounds[0] if self.colorbar.bounds else None,
            "vmax": self.colorbar.bounds[-1] if self.colorbar.bounds else None,
        }
        contourkwargs.update(
            {k: v for k, v in kwargs.items() if k not in self._sub_kwargs}
        )
        self._plot_handle = iplt.pcolormesh(data, **contourkwargs)


class __PPPlotHandlerMeta(abc.ABCMeta):
    def __new__(cls, name, bases, attrs):
        if name != "PPPlotHandlerABC" and "required_stash" not in attrs:
            raise AttributeError(
                "PlotHandler classes must have a 'required_stash' attribute"
            )
        newcls = super().__new__(cls, name, bases, attrs)
        return newcls

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        # Subclassing ABCMeta seems to break/disable the check
        # on whether all abstractmethods have been implemented
        # so do it manually.
        if name != "PPPlotHandlerABC" and cls.__abstractmethods__:
            raise TypeError(
                "{} has not implemented abstract methods: {}".format(
                    cls.__name__, ", ".join(cls.__abstractmethods__)
                )
            )


class PlotHandlerABC(abc.ABC):
    @abc.abstractmethod
    def plot(self):
        pass


class PPPlotHandlerABC(PlotHandlerABC, metaclass=__PPPlotHandlerMeta):
    pass


class MetDBPlotHandlerABC(PlotHandlerABC):
    pass
