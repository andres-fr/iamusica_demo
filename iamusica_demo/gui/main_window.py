#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module implements the main app widget that contains and manages all other
widgets, the ``MainWindow`` class.
"""


from PySide2 import QtCore, QtWidgets
#
from .spectrogram import SpecRollView
from .analysis_surface import AnalysisSurface
from .audio_player import AudioPlayer
#
from .core.widgets import NamedForm, IntSpinBox, DecimalSpinBox, PvalueSpinBox


# #############################################################################
# ## SANDBOX
# #############################################################################
class DetectorParameters(NamedForm):
    """
    Named form containing the frontend parameters for the detector.
    """
    NAME = "Detector parameters:"
    DETECTION_THRESHOLD_NAME = "threshold"
    #
    DETECTION_LPAD_NAME = "left pad"
    DETECTION_BODY_NAME = "detection chunks"
    DETECTION_RPAD_NAME = "right pad"
    #
    DETECTION_MEL_OFFSET_NAME = "dB gain"
    DETECTION_MEL_VSHIFT_NAME = "tuning shift"

    def __init__(self, parent=None, initial_pthresh=0.5,
                 initial_ovif=(10, 50, 10),
                 initial_mel_offset=0, initial_mel_vshift=0):
        """
        """
        super().__init__([
            (self.DETECTION_THRESHOLD_NAME,
             lambda default: PvalueSpinBox(
                 parent=None, default=initial_pthresh, step=0.01), None),
            #
            (self.DETECTION_LPAD_NAME,
             lambda default: IntSpinBox(
                 parent=None, minimum=0, default=initial_ovif[0]), None),
            #
            (self.DETECTION_BODY_NAME,
             lambda default: IntSpinBox(
                 parent=None, minimum=1, default=initial_ovif[1]), None),
            #
            (self.DETECTION_RPAD_NAME,
             lambda default: IntSpinBox(
                 parent=None, minimum=0, default=initial_ovif[2]), None),
            #
            (self.DETECTION_MEL_OFFSET_NAME,
             lambda default: DecimalSpinBox(
                 parent=None, default=initial_mel_offset, step=0.01), None),
            #
            (self.DETECTION_MEL_VSHIFT_NAME,
             lambda default: IntSpinBox(
                 parent=None, default=initial_mel_vshift), None)],
                         form_name=self.NAME,
                         parent=parent)


class AnalysisPan(QtWidgets.QSplitter):
    """
    """
    LOAD_AUDIO_BUTTON = "Append Audiofile"
    RECORD_BUTTON = "Record Audio"
    STOP_RECORD_BUTTON = "Stop Recording"
    RECORDING_COLOR = "  #EC7063"
    NOT_RECORDING_COLOR = " #82E0AA"
    REC_TEXT_COLOR = "#000000"

    def __init__(self, parent=None,
                 initial_pthresh=0.5, initial_ovif=(10, 50, 10),
                 initial_mel_offset=0, initial_mel_vshift=0,
                 player_delta_secs=1,
                 orientation=QtCore.Qt.Horizontal):
        """
        """
        super().__init__(parent=parent)
        self.setOrientation(orientation)
        # left side
        self.analface = AnalysisSurface(self)

        # right side: create HLayout for audio buttons
        self.load_b = QtWidgets.QPushButton(self.LOAD_AUDIO_BUTTON)
        self.record_b = QtWidgets.QPushButton()
        self.set_rec_but(recording=False)
        audio_buttons_lyt = QtWidgets.QHBoxLayout()
        audio_buttons_lyt.addWidget(self.load_b)
        audio_buttons_lyt.addWidget(self.record_b)
        # right side: create widgets for det params and player
        self.det_params = DetectorParameters(
            None, initial_pthresh, initial_ovif,
            initial_mel_offset, initial_mel_vshift)
        self.player = AudioPlayer(
            bw_delta_secs=player_delta_secs,
            fw_delta_secs=player_delta_secs)
        # put buttons, det params and player on VLayout
        right_lyt = QtWidgets.QVBoxLayout()
        right_lyt.addLayout(audio_buttons_lyt)
        right_lyt.addWidget(self.det_params)
        right_lyt.addWidget(self.player)
        # put right_lyt inside a widget and add everything to self
        right_wid = QtWidgets.QWidget()
        right_wid.setLayout(right_lyt)

        self.addWidget(self.analface)
        self.addWidget(right_wid)

    def set_rec_but(self, recording=True):
        """
        """
        if recording:
            self.record_b.setStyleSheet(
                f"background-color: {self.RECORDING_COLOR};" +
                f"color: {self.REC_TEXT_COLOR}")
            self.record_b.setText(self.STOP_RECORD_BUTTON)
        else:
            self.record_b.setStyleSheet(
                f"background-color: {self.NOT_RECORDING_COLOR};" +
                f"color: {self.REC_TEXT_COLOR}")
            self.record_b.setText(self.RECORD_BUTTON)

    def get_detector_params(self):
        """
        The DemoSession interface demands a parameterless function that
        returns ``(pthresh, ovif, mel_offset, mel_vshift)``. This method
        implemens that functionality.
        """
        state = self.det_params.get_state()
        #
        pthresh = state[self.det_params.DETECTION_THRESHOLD_NAME]
        lpad = state[self.det_params.DETECTION_LPAD_NAME]
        body = state[self.det_params.DETECTION_BODY_NAME]
        rpad = state[self.det_params.DETECTION_RPAD_NAME]
        offset = state[self.det_params.DETECTION_MEL_OFFSET_NAME]
        vshift = state[self.det_params.DETECTION_MEL_VSHIFT_NAME]
        #
        result = (pthresh, (lpad, body, rpad), offset, vshift)
        return result


# #############################################################################
# ## MAIN WINDOW
# #############################################################################
class IAMusicaMainWindow(QtWidgets.QMainWindow):
    """
    """

    SPEC_VFLIP = True

    def __init__(self, initial_display_width=1000,
                 initial_pthresh=0.5, initial_ovif=(10, 50, 10),
                 initial_mel_offset=0, initial_mel_vshift=0,
                 initial_spec_cmap="bone", initial_roll_cmap="cubehelix",
                 initial_vgrid_width=10,
                 font_size=12):
        """
        """
        super().__init__()
        # self._setup_undo()
        #
        # top side: spectrograms
        self.specroll = SpecRollView(self, initial_display_width,
                                     initial_spec_cmap, initial_roll_cmap,
                                     initial_vgrid_width)
        # bottom side: analysis pan
        self.analysis_pan = AnalysisPan(self, initial_pthresh, initial_ovif,
                                        initial_mel_offset, initial_mel_vshift)

        #
        # self.instructions_dialog = InstructionsDialog()
        # self.about_dialog = AboutDialog()

        # create main layout, add controller and graphics:
        self.main_splitter = QtWidgets.QSplitter()
        self.main_splitter.setOrientation(QtCore.Qt.Vertical)
        self.main_splitter.addWidget(self.specroll)
        self.main_splitter.addWidget(self.analysis_pan)
        #
        self.setCentralWidget(self.main_splitter)
        #
        self._setup_menu_bar()

    # def _setup_undo(self):
    #     """
    #     Set up undo stack and undo view
    #     """
    #     self.undo_stack = QtWidgets.QUndoStack(self)
    #     self.undo_view = QtWidgets.QUndoView(self.undo_stack)
    #     self.undo_view.setWindowTitle("Undo View")
    #     self.undo_view.setAttribute(QtCore.Qt.WA_QuitOnClose, False)

    def _setup_menu_bar(self):
        """
        Set up menu bar: create actions and connect them to methods.
        """
        # file menu
        edit_menu = self.menuBar().addMenu("Edit")
        self.create_action = edit_menu.addAction("Create session")
        self.open_action = edit_menu.addAction("Open session")
        # self.save_action = edit_menu.addAction("Save session")
        # help menu
        help_menu = self.menuBar().addMenu("Help")
        self.keyboard_shortcuts = help_menu.addAction("Keyboard shortcuts")
        # self.instructions = help_menu.addAction("Instructions")
        # self.instructions.triggered.connect(self.instructions_dialog.show)
        # self.about = help_menu.addAction("About")
        # self.about.triggered.connect(self.about_dialog.show)
