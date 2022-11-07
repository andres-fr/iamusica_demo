#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module contains an audio player widget, capable of playing audio from
numpy arrays, and including widgets for graphical interaction.
To use it, simply add it to your GUI and call ``player.set_array`` and
``player.configure_gui`` whenever a new array should be loaded.
"""


from PySide2 import QtCore, QtWidgets, QtMultimedia
#
from . import change_label_font, resize_button, seconds_to_timestamp
from . import QStream
from .core.widgets import DecimalSlider


# #############################################################################
# ## AUDIO PLAYER
# #############################################################################
class AudioPlayer(QtWidgets.QWidget):
    """
    This class contains a ``QMediaPlayer`` capable of playing numpy arrays
    representing audio files and loaded via the ``set_array`` method. Whenever
    an array is set, the ``configure_gui`` method should also be called.

    The class also implements interactive graphical elements to control
    playback (play/stop/etc buttons, volume slider...) as well as the logic
    to consistently bind those elements to the player.

    It abstracts all the machinery details so it can be simply added to the
    application and the interface remains simple but generic.
    """

    MAIN_PADDING = (3, 3, 3, 3)  # left, right, top, bottom (in pixels)
    SLIM_PADDING = (0, 0, 3, 3)
    #
    TIMER_WEIGHT = 50
    TIMER_SIZE = 7

    def __init__(self, parent=None, num_decimals=2,
                 bw_delta_secs=1, fw_delta_secs=1):
        """
        """
        self.num_decimals = num_decimals
        self.bw_delta_secs = bw_delta_secs
        self.fw_delta_secs = fw_delta_secs
        #
        super().__init__(parent)
        #
        self.play_icon = self.style().standardIcon(
            QtWidgets.QStyle.SP_MediaPlay)
        self.pause_icon = self.style().standardIcon(
            QtWidgets.QStyle.SP_MediaPause)
        self.stop_icon = self.style().standardIcon(
            QtWidgets.QStyle.SP_MediaStop)
        self.bw_icon = self.style().standardIcon(
            QtWidgets.QStyle.SP_MediaSeekBackward)
        self.fw_icon = self.style().standardIcon(
            QtWidgets.QStyle.SP_MediaSeekForward)
        # first create player, then setup GUI, finally connect player to GUI
        self.media_player = QtMultimedia.QMediaPlayer(parent=self)
        self.media_player.setAudioRole(QtMultimedia.QAudio.MusicRole)
        # audio probe provides "real-time" readings from the source (the player)
        self.probe = QtMultimedia.QAudioProbe(parent=self)
        self.probe.setSource(self.media_player)
        self.probe.audioBufferProbed.connect(self.on_audio_probe)
        #
        self._setup_gui()
        self.media_player.stateChanged.connect(self.on_media_state_changed)
        # self.media_player.positionChanged.connect(self.on_pos_player_update)
        self.media_player.positionChanged.connect(
            lambda pos: self.on_pos_update(pos, "player"))
        self.media_player.error.connect(self.on_media_error)
        # this is needed to break the slider<->player update loop
        self._block_slider_player_loop = False
        # initialize with disabled buttons and zero range
        self.configure_gui(min_val=0, max_val=0, step=0.1, pos_secs=0,
                           enable_buttons=False)
        # call this function to set the play/pause icon to media_player state
        self.on_media_state_changed()
        self.vol_s.setValue(50)  # also initialize volume
        #
        self.current_qstream = None

    def _setup_gui(self):
        """
        """
        # create buttons
        self.play_b = QtWidgets.QPushButton()
        self.play_b.clicked.connect(self.on_play)
        self.stop_b = QtWidgets.QPushButton()
        self.stop_b.setIcon(self.stop_icon)
        self.stop_b.clicked.connect(self.media_player.stop)
        self.bw_b = QtWidgets.QPushButton()
        self.bw_b.setIcon(self.bw_icon)
        self.bw_b.clicked.connect(lambda: self.delta_pos(-self.bw_delta_secs))
        self.fw_b = QtWidgets.QPushButton()
        self.fw_b.setIcon(self.fw_icon)
        self.fw_b.clicked.connect(lambda: self.delta_pos(self.fw_delta_secs))
        # configure button sizes
        resize_button(self.play_b, padding_px_lrtb=self.MAIN_PADDING)
        resize_button(self.stop_b, padding_px_lrtb=self.MAIN_PADDING)
        resize_button(self.bw_b, w_ratio=0.8, padding_px_lrtb=self.SLIM_PADDING)
        resize_button(self.fw_b, w_ratio=0.8, padding_px_lrtb=self.SLIM_PADDING)
        # create volume slider
        self.vol_s = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.vol_s.setRange(0, 100)
        max_play_h = max([sz.height() for sz in self.play_icon.availableSizes()])
        self.vol_s.setMaximumHeight(max_play_h)
        self.vol_s.valueChanged.connect(self.on_vol_change)
        # create position slider and label
        self.pos_s = DecimalSlider(
            0, 0, self.num_decimals, QtCore.Qt.Horizontal)
        self.pos_l = QtWidgets.QLabel()
        change_label_font(self.pos_l, self.TIMER_WEIGHT, self.TIMER_SIZE)
        # self.pos_s.decimalValueChanged.connect(self.on_pos_slider_update)
        self.pos_s.decimalValueChanged.connect(
            lambda: self.on_pos_update(self.pos_s.value(), "slider"))

        # add widgets to layout
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.addWidget(self.play_b)
        self.main_layout.addWidget(self.bw_b)
        self.main_layout.addWidget(self.stop_b)
        self.main_layout.addWidget(self.fw_b)
        self.main_layout.addWidget(self.vol_s)
        self.main_layout.addWidget(self.pos_s)
        self.main_layout.addWidget(self.pos_l)

    # play/pause functionality
    def is_playing(self):
        """
        """
        x = self.media_player.state() == QtMultimedia.QMediaPlayer.PlayingState
        return x

    def on_play(self):
        """
        When the play/pause button is clicked, simply pass on the action to the
        media player.
        """
        if self.is_playing():
            self.media_player.pause()
        else:
            self.media_player.play()

    def on_media_state_changed(self, *args):
        """
        Every time the media player updates state, update the play/pause icon.
        """
        if self.is_playing():
            self.play_b.setIcon(self.pause_icon)
        else:
            self.play_b.setIcon(self.play_icon)

    # timer/position functionality. This one is a bit trickier: moving the
    # slider always updates the player and the label. But when player position
    # changes, this also "updates" the slider. Fortunately this is not an
    # infinite loop since the slider value only updates when changed.
    def get_pos(self):
        """
        """
        mp_pos = self.media_player.position()
        pos_secs = round(mp_pos / 1000, self.num_decimals)
        return pos_secs

    def set_pos(self, pos_secs):
        """
        """
        mp_pos = pos_secs * 1000
        self.media_player.setPosition(mp_pos)

    def delta_pos(self, delta=0):
        """
        """
        pos_secs = self.get_pos()
        new_pos = pos_secs + delta
        new_pos = max(new_pos, 0)
        new_pos = min(new_pos, self.pos_s.getMaximum())
        self.set_pos(new_pos)

    def on_pos_update(self, pos, caller):
        """
        """
        if caller == "slider" and not self._block_slider_player_loop:
            self._block_slider_player_loop = True
            pos_secs = pos
            self.set_pos(pos_secs)
            self.set_time_label(pos_secs)
            self._block_slider_player_loop = False
        elif caller == "player" and not self._block_slider_player_loop:
            self._block_slider_player_loop = True
            # SI ESTO CORRE, ACTIVAR UN FLAG!
            pos_secs = round(float(pos) / 1000, self.num_decimals)
            self.pos_s.setValue(pos_secs)
            self.set_time_label(pos_secs)
            self._block_slider_player_loop = False

    def set_time_label(self, seconds):
        """
        """
        pos_txt = seconds_to_timestamp(seconds, self.num_decimals)
        self.pos_l.setText(pos_txt)

    def on_media_status_changed(self, status):
        """
        status: elements from the player.MediaStatus enumeration (LoadingMedia,
        LoadedMedia...)
        """
        if status == self.media_player.LoadedMedia:
            md_keys = self.media_player.availableMetaData()
            for k in md_keys:
                print(self.media_player.metaData(k))
        else:
            # disable all buttons
            pass

    # def set_media_stream(self, arr, samplerate):
    def set_array(self, arr, samplerate):
        """
        :param arr: A float numpy array of rank 1
        :param samplerate: In Hz

        Given an audio array and its samplerate, converts it to a QStream and
        sets this player's media to the stream. Automatically handles opening
        and closing of streams.


        :param qstream: A ``QtCore.QBuffer()`` containing the wav data. It can
          be obtained from a numpy array via ``wavarr_to_qstream(arr, sr)``

        Closes current stream if exists, then opens the given ``qstream`` in
        the given ``mode``, and calls ``setMedia`` on the given  ``qstream``.
        """
        if self.current_qstream is not None:
            self.current_qstream.close()
        self.current_qstream = QStream(arr, samplerate)
        self.current_qstream.open()
        self.media_player.setMedia(QtMultimedia.QMediaContent(),
                                   self.current_qstream)

    # miscellaneous
    def on_vol_change(self, vol):
        """
        """
        self.media_player.setVolume(vol)

    def configure_gui(self, min_val=0, max_val=0, step=0.1, pos_secs=0,
                      enable_buttons=True):
        """
        Enable/disable buttons, configure position slider range+step, and set
        player position.
        """
        self.pos_s.setMinimum(min_val)
        self.pos_s.setMaximum(max_val)
        self.pos_s.setSingleStep(step)
        self.set_pos(pos_secs)  # this updates label as well
        # self.set_time_label(self.get_pos())
        self.play_b.setEnabled(enable_buttons)
        self.stop_b.setEnabled(enable_buttons)
        self.bw_b.setEnabled(enable_buttons)
        self.fw_b.setEnabled(enable_buttons)

    def on_media_error(self, err):
        """
        """
        raise Exception(f"Media error! {err}\n" +
                        "Is array well formatted? All dependencies installed?")

    def on_audio_probe(self, audio_buffer):
        """
        ATM not working for PySide2:
        stackoverflow.com/questions/66417106/reading-qaudioprobe-buffer
        bugreports.qt.io/projects/PYSIDE/issues/PYSIDE-934?filter=allissues
        """
        # print("buffer:", audio_buffer.frameCount())
        # print(audio_buffer.data().toBytes())
        pass
