#!/usr/bin python
# -*- coding:utf-8 -*-


"""
This module contains functionality to handle the real-time input audio process.
"""


import numpy as np
import pyaudio


# ##############################################################################
# # AUDIO INPUT STREAM (ASYNCH LOOP)
# ##############################################################################
class AsynchAudioInputStream:
    """
    This class holds and manages a pyaudio stream to record audio and
    periodically call a callback function when audio packets arrive.

    Use the OS audio settings to select the active microphone.
    """

    IN_CHANNELS = 1
    PYAUDIO_DTYPE = pyaudio.paFloat32
    NP_DTYPE = np.float32

    def __init__(self, samplerate=16000, chunk_length=1024,
                 update_callback=None):
        """
        :param update_callback: Optional. A function that accepts the current
          update as 1D numpy array, to be called after each update. Return
          value ignored.
        """
        self.sr = samplerate
        self.chunk = chunk_length
        # setup recording stream
        self.pa = pyaudio.PyAudio()

        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)
            if devinfo["maxInputChannels"] > 0:
                try:
                    self.stream = self.pa.open(format=self.PYAUDIO_DTYPE,
                                               channels=self.IN_CHANNELS,
                                               rate=samplerate,
                                               input_device_index=i,
                                               input=True,  # record
                                               output=False,  # playback
                                               frames_per_buffer=chunk_length,
                                               stream_callback=self.callback,
                                               start=False)
                    print("\n\nsupported:", devinfo)
                    break
                except (ValueError, OSError):  # as e:
                    # print("\n\nAudio device issues!", type(e), e)
                    pass
        #
        self.update_callback = update_callback

    def read(self):
        """
        Returns the current reading from the ring buffer, unwrapped so
        that the first element is the oldest.
        """
        return self.rb.read()

    def start(self):
        """
        Starts updating the ring buffer with readings from the microphone.
        """
        self.stream.start_stream()

    def stop(self):
        """
        Stops updating the ring buffer (but doesn't delete its contents).
        """
        self.stream.stop_stream()

    def terminate(self):
        """
        Close the input stream (can't record afterwards).
        """
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    def __enter__(self):
        """
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        """
        self.terminate()

    def callback(self, in_data, frame_count, time_info, status):
        """
        This function is automatically called by ``self.p`` every time there is
        new recorded data. By convention it returns the buffer plus a flag.

        :param in_data: Recorded data as bytestring as ``cls.PYAUDIO_DTYPE``
        :param frame_count: Number of samples in recorded data (``self.chunk``)
        :param time_info: unused
        :param status: unused
        """
        in_arr = np.frombuffer(in_data, dtype=self.NP_DTYPE)
        if self.update_callback is not None:
            self.update_callback(in_arr)
        return (in_arr, pyaudio.paContinue)
