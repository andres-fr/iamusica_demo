#!/usr/bin python
# -*- coding:utf-8 -*-


"""
"""


import os
import sys
import datetime
import pytz
import logging
from typing import Optional
import socket
#
#
# ignore "mypy" import error for coloredlogs.
# see https://mypy.readthedocs.io/en/latest/running_mypy.html#missing-imports
import coloredlogs  # type: ignore
import torchaudio


# ##############################################################################
# # I/O
# ##############################################################################
def torch_load_resample_audio(path, target_sr=16000, mono=True,
                              normalize=True, device="cpu"):
    """
    Analogously to ``librosa.load``, this function loads and resamples a wav
    file. The resampling operation from torchaudio is much faster.
    :param path: Absolute path to the wav file to be loaded
    :param target_sr: Returned wavfile will have this sample rate
    :param mono: If true, returned wavfile will be averaged down to mono.
    :param device: Returned wavfile will be on the specified device.
    """
    wave, sr_in = torchaudio.load(path, normalize=normalize)
    resampler = torchaudio.transforms.Resample(sr_in, target_sr).to(device)
    if mono:
        wave = wave.mean(dim=0)
    wave = resampler(wave.to(device))
    return wave

def torch_save_wav(t, outpath, samplerate,
                   encoding="PCM_S", bits_per_sample=16):
    """
    :param t: Rank-1 float tensor containing an audio waveform to be saved
    """
    torchaudio.save(outpath, t, samplerate,
                    encoding="PCM_S", bits_per_sample=16)


# ##############################################################################
# # STRING
# ##############################################################################
def make_timestamp(timezone="Europe/Berlin", with_tz_output=False):
    """
    Output example: day, month, year, hour, min, sec, milisecs:
    10_Feb_2018_20:10:16.151
    """
    ts = datetime.datetime.now(tz=pytz.timezone(timezone)).strftime(
        "%Y_%m_%d_%H_%M_%S.%f")[:-3]
    if with_tz_output:
        return "%s(%s)" % (ts, timezone)
    else:
        return ts


# ##############################################################################
# # COLOR LOGGER
# ##############################################################################
class HostnameFilter(logging.Filter):
    """
    Needed to include hostname into the logger. See::
      https://stackoverflow.com/a/55584223/4511978
    """

    def filter(self, record) -> bool:
        record.hostname = socket.gethostname()
        return True


class ColorLogger:
    """
    This class:
    1. Creates a ``logging.Logger`` with a convenient configuration.
    2. Attaches ``coloredlogs.install`` to it for colored terminal output
    3. Provides some wrapper methods for convenience
    Usage example::
      # create 2 loggers
      cl1 = ColorLogger("term.and.file.logger", "/tmp/test.txt")
      cl2 = ColorLogger("JustTermLogger")
      # use them at wish
      cl1.logger.debug("this is a debugging message")
      cl2.logger.info("this is an informational message")
      cl1.logger.warning("this is a warning message")
      cl2.logger.error("this is an error message")
      cl1.logger.critical("this is a critical message")
    """

    FORMAT_STR = ("%(asctime)s.%(msecs)03d %(hostname)s: %(name)s" +
                  "[%(process)d] %(levelname)s %(message)s")

    def get_logger(self, logger_name, logfile_dir: Optional[str],
                   filemode: str = "a",
                   logging_level: int = logging.DEBUG) -> logging.Logger:
        """
        :param filemode: In case ``logfile_dir`` is given, this specifies the
          output mode (e.g. 'a' for append).
        :returns: a ``logging.Logger`` configured to output all events at level
          ``self.logging_level`` or above into ``sys.stdout`` and (optionally)
          the given ``logfile_dir``, if not None.
        """
        # create logger, formatter and filter, and set desired logger level
        logger = logging.getLogger(logger_name)
        formatter = logging.Formatter(self.FORMAT_STR,
                                      datefmt="%Y-%m-%d %H:%M:%S")
        hostname_filter = HostnameFilter()
        logger.setLevel(logging_level)
        # create and wire stdout handler
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.addFilter(hostname_filter)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
        # optionally, create and wire file handler
        if logfile_dir is not None:
            basename = make_timestamp(
                with_tz_output=False) + logger_name + ".log"
            logfile_path = os.path.join(logfile_dir, basename)
            # create one handler for print and one for export
            file_handler = logging.FileHandler(logfile_path, filemode)
            file_handler.addFilter(hostname_filter)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        #
        return logger

    def __init__(self, logger_name: str,
                 logfile_path: Optional[str] = None,
                 filemode: str = "a",
                 logging_level: int = logging.DEBUG):
        """
        :param logger_name: A process may have several loggers. This parameter
          distinguishes them.
        :param logfile_path: Where to write out.
        """
        self.logger: logging.Logger = self.get_logger(logger_name,
                                                      logfile_path, filemode,
                                                      logging_level)
        #
        coloredlogs.install(logger=self.logger,
                            fmt=self.FORMAT_STR,
                            level=logging_level)

    # a few convenience wrappers:
    def debug(self, *args, **kwargs) -> None:
        self.logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs) -> None:
        self.logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs) -> None:
        self.logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs) -> None:
        self.logger.error(*args, **kwargs)

    def critical(self, *args, **kwargs) -> None:
        self.logger.critical(*args, **kwargs)
