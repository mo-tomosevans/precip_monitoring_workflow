from typing import Union, List
from pathlib import Path
from abc import abstractmethod
import tempfile
import shutil
import multiprocessing
import tarfile
import logging
from oemplotlib import LOGGER as _OEMLOGGER

LOGGER = _OEMLOGGER.getChild(__name__)
# LOGGER.setLevel(logging.DEBUG)

_DEFAULT_TAR_NAME = "plots.tar"


class SaveHandlerABC:
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def get_working_fname(self, fname: Union[str, Path]):
        pass

    @abstractmethod
    def get_imt_working_dir(self, imt_dir: Union[str, Path]):
        pass

    @abstractmethod
    def handle_output_files(self, filename_list: List[Union[str, Path]]):
        pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass


class NullHandler(SaveHandlerABC):
    """A SaveHandler that does nothing"""

    def get_working_fname(self, fname: Union[str, Path]):
        return fname

    def get_imt_working_dir(self, imt_dir: Union[str, Path]):
        return imt_dir

    def handle_output_files(self, filename_list: List[Union[str, Path]]):
        return filename_list


class _TmpdirMixin:
    """Provides a context allowing for plotting to take place in temporary directories
    """

    def __init__(self, tmpdir_root: Path = None, **kwargs) -> None:
        """Set up the temporary directories

        Args:
            tmpdir_root (Path, optional): The directory under which all other
                                          temporary and/or working directories
                                          should be created. Defaults to None.

        Raises:
            ValueError: Raised if tmpdir_root is not set
        """

        if tmpdir_root is None:
            raise ValueError("tmpdir_root must be specified")
        self._tmpdir_root = tmpdir_root

        self._tmp_dir = None

        super().__init__(**kwargs)

    def __enter__(self):
        super().__enter__()
        self._tmp_dir = tempfile.TemporaryDirectory(dir=self._tmpdir_root)
        if self._working_dir is None:
            self._working_dir = Path(self._tmp_dir.name)
        self._working_dir.mkdir(exist_ok=True, parents=True)

    def __exit__(self, type, value, traceback):
        self._tmp_dir.cleanup()
        super().__exit__(type, value, traceback)


class _IMTMixin:
    """Provides a context that correctly handles IMT databases when working in a temporary directory
    """

    def __init__(
        self, output_dir: Path = None, imt_db_dir: Path = None, **kwargs
    ) -> None:
        self._output_dir = output_dir
        self._imt_db_dir = imt_db_dir
        self._imt_working_dir = None
        super().__init__(**kwargs)

    def __enter__(self):
        super().__enter__()
        if getattr(self, "_working_dir", None) is None:
            raise ValueError("unable to set IMT dirs unless working_dir is set")
        self._imt_working_dir = self._working_dir
        self._working_dir = self._working_dir / self._output_dir.resolve().relative_to(
            self._imt_db_dir.resolve()
        )
        self._working_dir.mkdir(exist_ok=True, parents=True)

    def __exit__(self, type, value, traceback):
        LOGGER.info("Starting copying db files back to main directory")
        imt_dbs = self._imt_working_dir.glob("*.db")
        for db in imt_dbs:
            shutil.copy(db, self._imt_db_dir)
        LOGGER.info("Finished copy db files back to main directory.")
        super().__exit__(type, value, traceback)

    def get_imt_working_dir(self) -> Path:
        """Returns the temporary directory that will be used for IMT databases during plotting

        Raises:
            ValueError: Raised if the IMT directory has not been set. Usually
                        this means that the class is not being used as a context manager.

        Returns:
            Path: The path of the IMT working directory
        """
        if self._imt_working_dir is None:
            raise ValueError("_imt_working_dir not set, is the context open?")
        return self._imt_working_dir


class TarHandler(_IMTMixin, _TmpdirMixin, SaveHandlerABC):
    """A save handler that save plots in to a tar file.
    
    This handler redirects the plotting output to a working directory
    and sets up a background process to add plots to a tar file
    as they are created.

    .. note ::
       This class should be used as a context manager. The context will
       set up and clean up the background process.
    """

    class _ProcessKiller:
        """A class that can be used to stop a queue"""

    def __init__(
        self,
        tarfile_dir: Path = None,
        working_dir: Path = None,
        tarfile_name: str = _DEFAULT_TAR_NAME,
        **kwargs
    ) -> None:
        """Initialise TarHandler

        Args:
            tarfile_dir (Path, optional): Path to the directory where the output
                                          tar file should be created. Defaults to None.
            working_dir (Path, optional): Directory where plots should be stored until they
                                          have been added to the tar file. Defaults to None.
            tarfile_name (str, optional): The name of the tar file. Defaults to _DEFAULT_TAR_NAME.

        Raises:
            ValueError: Raised if tarfile_dir or working_dir are not specified
        """

        self._tarfile_name = tarfile_name
        self._tarfile_dir = tarfile_dir
        self._tarfile = None

        self._working_dir = working_dir

        LOGGER.debug(
            "TarHandler initialised for tar file %s %s using working dir %s",
            self._tarfile_dir,
            self._tarfile_name,
            self._working_dir,
        )

        self._process = None
        self._queue = None

        super().__init__(**kwargs)

    def _consumer(self):
        """Function to run in a background process.
        
        This functions runs in a background process adding plots to the tarfile
        and removing them from the working directory.
        """

        LOGGER.debug("TarHandler consumer started")
        tarfile_full_path = self._tarfile_dir / self._tarfile_name
        self._tarfile = tarfile.open(tarfile_full_path, "w")
        try:
            while True:
                try:
                    filelist = self._queue.get(block=True)

                    if isinstance(filelist, self._ProcessKiller):
                        # poison pill, finish processing
                        break

                    if isinstance(filelist, str):
                        filelist = [filelist]
                    LOGGER.debug("TarHandler consumer handling %s items", len(filelist))
                    for name in filelist:
                        LOGGER.debug(
                            "TarHandler adding %s to tar file %s",
                            name,
                            tarfile_full_path,
                        )
                        try:
                            arcname = str(Path(name).name)
                            self._tarfile.add(name, arcname=arcname)
                            Path(name).unlink()
                        except Exception:
                            LOGGER.exception(
                                "TarHandler error adding file %s to tar with name %s",
                                name,
                                arcname,
                            )
                except Exception:
                    LOGGER.exception(
                        "TarHandler unknown error handling item from queue"
                    )
        finally:
            self._tarfile.close()

    def __enter__(self):
        super().__enter__()
        if self._tarfile_dir is None:
            self._tarfile_dir = self._output_dir
        self._queue = multiprocessing.Queue()
        self._process = multiprocessing.Process(target=self._consumer)
        self._process.start()
        return self

    def __exit__(self, type, value, traceback):
        if self._process is not None:
            self.stop_consumer()
            self._process.join()
            self._process.close()
        return super().__exit__(type, value, traceback)

    def get_working_fname(self, fname: Union[str, Path]) -> Union[str, Path]:
        """Redirects output from the plotting routine to the working directory

        Args:
            fname (Union[str, Path]): input file name

        Returns:
            Union[str, Path]: file name within the working directory
        """

        return_str = False
        if isinstance(fname, str):
            fname = Path(fname)
            return_str = True
        filename = self._working_dir / fname.name
        LOGGER.debug("TarHandler calculated working name %s", filename)
        if return_str:
            return str(filename)
        return filename

    def handle_output_files(
        self, filename_list: List[Union[str, Path]]
    ) -> List[Union[str, Path]]:
        """Adds files names to a queue so the background process will put them in the tar file

        Args:
            filename_list (List[Union[str, Path]]): List of files to be added to the tar file

        Raises:
            ValueError: Raised if the queue has not been opened. Usually this means the instance
                        is not being used as a context manager.

        Returns:
            List[Union[str, Path]]: List of files that should have been added to the tar file.
                                    .. warning ::
                                    This does not guarantee that the files have been added to the tar file,
                                    only that they have been passed to the background process for processing.
        """

        if self._queue is None:
            raise ValueError(
                "queue not initialised, TarHandler should be used as a context manager"
            )
        self._queue.put(filename_list)

        # files were successfully passed to background process so return the file names
        return filename_list

    def stop_consumer(self):
        """Stop the background process

        Raises:
            ValueError: Raised if the background process has not been started
        """

        if self._queue is None:
            raise ValueError(
                "queue not initialised, TarHandler should be used as a context manager"
            )
        self._queue.put(self._ProcessKiller())
