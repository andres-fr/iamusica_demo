# !/usr/bin/env python
# -*- coding:utf-8 -*-


"""
python -m iamusica_demo
"""


import os
import sys
# For omegaconf
from dataclasses import dataclass
from typing import Optional, List
#
from omegaconf import OmegaConf, MISSING
# Dark theme: https://qdarkstylesheet.readthedocs.io/en/latest/readme.html
import numpy as np
import qdarkstyle
from PySide2 import QtWidgets, QtGui, QtCore
# app constants
from . import ASSETS_PATH, OV_MODEL_PATH, OV_MODEL_CONV1X1_HEAD, \
    OV_MODEL_LRELU_SLOPE, WAV_SAMPLERATE, NUM_PIANO_KEYS
from . import MEL_FRAME_SIZE, MEL_FRAME_HOP, NUM_MELS, MEL_FMIN, MEL_FMAX, \
    MEL_WINDOW
# app backend
from .utils import make_timestamp
from .models import TorchWavToLogmelDemo, get_ov_demo_model
from .session import SessionHDF5, DemoSession
# app frontend
from .gui.main_window import IAMusicaMainWindow
from .gui.core.dialogs import FlexibleDialog, ExceptionDialog, InfoDialog



# ##############################################################################
# # IAMUSICA APP
# ##############################################################################
class QtDemoSession(QtCore.QObject, DemoSession):
    """
    DemoSession with a signal
    :cvar: When recording new chunks, ``wav_update_callback`` sends a signal to
      the main app, to update the GUI. This is a string message signaling where
      to re-position the spectrograms after a new chunk has been recorded.
    """
    recInferenceSignal = QtCore.Signal(str)
    REC_XPOS = "end"

    def __init__(self, *demo_args, **demo_kwargs):
        """
        """
        QtCore.QObject.__init__(self)
        main_app = demo_kwargs.pop("main_app")
        DemoSession.__init__(self, *demo_args, **demo_kwargs)
        self.recInferenceSignal.connect(main_app.update_frontend_from_session)

    def wav_update_callback(self, chunk):
        """
        """
        did_inference = super().wav_update_callback(chunk)
        if did_inference:
            self.recInferenceSignal.emit(self.REC_XPOS)


class NoSessionDialog(InfoDialog):
    """
    For session-related functionality, whenever no session is present.
    """
    def __init__(self, timeout_ms=500):
        """
        :param save_dict: A dict with ``item_name: save_path`` pairs.
        """
        super().__init__("NO SESSION", "Please create/load session first!",
                         timeout_ms=timeout_ms)


class RecDialog(InfoDialog):
    """
    For session-related functionality, whenever session is currently recording
    """
    def __init__(self, timeout_ms=500):
        """
        :param save_dict: A dict with ``item_name: save_path`` pairs.
        """
        super().__init__("RECORDING IN PROGRESS",
                         "Please finish recording first!",
                         timeout_ms=timeout_ms)



class SavedInfoDialog(InfoDialog):
    """
    Informative dialog telling about saved paths.
    """
    def __init__(self, save_dict, timeout_ms=500):
        """
        :param save_dict: A dict with ``item_name: save_path`` pairs.
        """
        super().__init__("SAVED", self.save_dict_to_str(save_dict),
                         timeout_ms=timeout_ms,
                         header_style="font-weight: bold; color: green")

    @staticmethod
    def save_dict_to_str(save_dict):
        """
        """
        msg = "\n".join(["Saved {} to {}".format(k, v)
                         for k, v in save_dict.items()])
        return msg


class SaveWarningDialog(InfoDialog):
    """
    A dialog to be prompted when trying to delete unsaved changes.
    Usage example::

      self.dialog = SaveWarningDialog()
      user_wants_to_remove = bool(self.dialog.exec_())
      ...
    """

    def __init__(self):
        """
        """
        super().__init__("WARNING",
                         "Close current session?",
                         "YES", "NO",
                         print_msg=False)
        self.reject_b.setDefault(True)


class KeymapsDialog(FlexibleDialog):
    """
    Info dialog showing keymap list
    """
    def __init__(self, mappings, parent=None):
        """
        """
        self.mappings = mappings
        super().__init__(parent=parent)

    def setup_ui_body(self, widget):
        """
        """
        lyt = QtWidgets.QVBoxLayout(widget)
        #
        self.list_widget = QtWidgets.QListWidget()
        for k, v in self.mappings.items():
            self.list_widget.addItem("{} ({})".format(k, v))
        lyt.addWidget(self.list_widget)


class IAMusicaApp(QtWidgets.QApplication):
    """
    """
    INFO_DIALOG_TIMEOUT_MS = 1500
    TORCH_DEVICE = "cpu"

    H5_WAV_FSTRING = "wav_{}.h5"
    H5_MEL_FSTRING = "mel_{}.h5"
    H5_ONSET_FSTRING = "onset_{}.h5"

    LOAD_AUDIO_FILTER = "WAV files (*.wav *.WAV)"

    KEYMAPS = {
        # "Undo": QtGui.QKeySequence("Ctrl+Z"),
        # "Redo": QtGui.QKeySequence("Ctrl+Y"),
        # "View undo stack": QtGui.QKeySequence("Alt+Z"),
        #
        "Create session": QtGui.QKeySequence("Ctrl+N"),
        "Open session": QtGui.QKeySequence("Ctrl+O"),
        "Save session": QtGui.QKeySequence("Ctrl+S"),
        ### "Quicksave text": QtGui.QKeySequence("Ctrl+Shift+S"),
        #
        "Toggle play/pause": QtGui.QKeySequence("Ctrl+Space"),
        "Seek player back <<": QtGui.QKeySequence("Ctrl+Left"),
        "Seek player forward >>": QtGui.QKeySequence("Ctrl+Right"),
        "Start/Stop recording": QtGui.QKeySequence("Ctrl+R"),
        #
        "Zoom in": QtGui.QKeySequence("Ctrl+Up"),
        "Zoom out": QtGui.QKeySequence("Ctrl+Down")
    }

    def __init__(self,
                 # frontend
                 app_name="IAMusica Demo", initial_display_width=800,
                 # backend
                 wav_samplerate=16000,
                 mel_frame_size=2048,
                 mel_frame_hop=384, num_mels=229, mel_fmin=50, mel_fmax=8000,
                 mel_window=MEL_WINDOW, ov_model_path=OV_MODEL_PATH,
                 ov_model_conv1x1_head=OV_MODEL_CONV1X1_HEAD,
                 ov_model_lrelu_slope=OV_MODEL_LRELU_SLOPE,
                 num_piano_keys=NUM_PIANO_KEYS,
                 # demo
                 audio_recording_numhops=4, h5_chunk_numhops=60,
                 initial_pthresh=0.5,  initial_ovif=(10, 50, 10),
                 initial_mel_offset=0, initial_mel_vshift=0,
                 initial_spec_cmap="bone", initial_roll_cmap="cubehelix",
                 initial_vgrid_width=10,
                 zoom_percentage=5,
                 #
                 num_analysis_histbins=50,
                 workspace_dir=None):
        """
        :param audio_recording_numhops: Audio acquisition comes in this many
          chunks every time. Each chunk consists of ``hopsize`` samples.
        :param h5_chunk_numhops: Each read/write operation to the HDF5 files
          will at least involve this many chunks. If too slow or too large,
          read/write will be slower, but still correct. Ideally it should be
          slightly larger than the average read/write operation.
        """
        super().__init__([app_name])
        # frontend
        self.workspace_dir = (os.path.expanduser("~") if workspace_dir is None
                              else workspace_dir)
        self.load_audio_dir = (os.path.expanduser("~") if workspace_dir is None
                               else workspace_dir)
        self.keymaps_dialog = KeymapsDialog(
            {k: v.toString() for k, v in self.KEYMAPS.items()})

        self.main_window = IAMusicaMainWindow(
            initial_display_width, initial_pthresh, initial_ovif,
            initial_mel_offset, initial_mel_vshift,
            initial_spec_cmap, initial_roll_cmap, initial_vgrid_width)
        # backend
        self.logmel_fn = TorchWavToLogmelDemo(
            wav_samplerate, mel_frame_size, mel_frame_hop, num_mels,
            mel_fmin, mel_fmax, mel_window)
        self.ov_model = get_ov_demo_model(
            ov_model_path, num_mels, num_piano_keys,
            ov_model_conv1x1_head, ov_model_lrelu_slope, self.TORCH_DEVICE)
        #
        self.wav_samplerate = wav_samplerate

        # session
        self.audio_recording_numhops = audio_recording_numhops
        self.h5_chunk_numhops = h5_chunk_numhops
        self.num_mels = num_mels
        self.num_piano_keys = num_piano_keys
        self.session = None
        #
        self.zoom_percentage = zoom_percentage
        #
        self.num_analysis_histbins = num_analysis_histbins
        #
        self.connect_frontend_and_backend()

    def connect_frontend_and_backend(self):
        """
        """
        self.main_window.open_action.triggered.connect(self.load_session)
        self.main_window.create_action.triggered.connect(self.create_session)
        #
        self.main_window.keyboard_shortcuts.triggered.connect(
            self.keymaps_dialog.show)
        # connect load/record buttons
        self.main_window.analysis_pan.load_b.pressed.connect(
            self.load_audio_file)
        self.main_window.analysis_pan.record_b.pressed.connect(
            self.record_audio)

        # menu bar shortcuts
        self.main_window.create_action.setShortcut(
            self.KEYMAPS["Create session"])
        self.main_window.open_action.setShortcut(self.KEYMAPS["Open session"])
        # player/rec shortcuts
        player = self.main_window.analysis_pan.player
        QtWidgets.QShortcut(self.KEYMAPS["Toggle play/pause"],
                            self.main_window, player.play_b.click)
        QtWidgets.QShortcut(self.KEYMAPS["Seek player back <<"],
                            self.main_window, player.bw_b.click)
        QtWidgets.QShortcut(self.KEYMAPS["Seek player forward >>"],
                            self.main_window, player.fw_b.click)
        QtWidgets.QShortcut(self.KEYMAPS["Start/Stop recording"],
                            self.main_window,
                            self.main_window.analysis_pan.record_b.click)
        # spectrogram shortcuts
        QtWidgets.QShortcut(
            self.KEYMAPS["Zoom in"], self.main_window,
            lambda: self.main_window.specroll.zoom(-self.zoom_percentage))
        QtWidgets.QShortcut(
            self.KEYMAPS["Zoom out"], self.main_window,
            lambda: self.main_window.specroll.zoom(self.zoom_percentage))
        # connect roll view to analysis surface
        roll_view = self.main_window.specroll.roll_view

        roll_view.selRectSignal.connect(self.update_analysis_selrect)
        roll_view.velGridSignal.connect(self.update_analysis_vgrid)

    @staticmethod
    def new_session_name():
        """
        """
        ts = make_timestamp(timezone="Europe/Berlin", with_tz_output=False)
        return ts

    def delete_existing_session_if_exists(self):
        """
        :returns: true if no pre-existing session, or if pre-existing session
          can be deleted. False only if session pre-exists and user backed out.

        """
        result = True
        if self.session is not None:
            self.delete_sess_dialog = SaveWarningDialog()
            delete_sess = bool(self.delete_sess_dialog.exec_())
            if delete_sess:
                self.session.terminate()
                self.session = None
                self.main_window.specroll.clear()
            else:
                # if user backs out, keep current session
                result = False
        return result

    @classmethod
    def process_session_dir(cls, sess_dir, err_if_missing=True):
        """
        :param err_if_missing: If true, raises an ``AssertionError`` if any of
          the returned HDF5 files doesn't already exist. Set to true to check
          if a dir is a session, set to false to create a new session.
        :returns: The paths for the wav, mel and onset HDF5 files inside the
          givne directory.
        :raises: Error if any of the files doesn't exist and ``err_if_missing``.
        """
        ws_dir, sess_name = os.path.split(sess_dir)
        ws_dir = os.path.join(ws_dir, sess_name)
        h5w_path = os.path.join(ws_dir, cls.H5_WAV_FSTRING.format(sess_name))
        h5m_path = os.path.join(ws_dir, cls.H5_MEL_FSTRING.format(sess_name))
        h5o_path = os.path.join(ws_dir, cls.H5_ONSET_FSTRING.format(sess_name))
        #
        if err_if_missing:
            assert os.path.exists(h5w_path), f"{h5w_path} doesn't exist!"
            assert os.path.exists(h5m_path), f"{h5m_path} doesn't exist!"
            assert os.path.exists(h5o_path), f"{h5o_path} doesn't exist!"
        return (h5w_path, h5m_path, h5o_path)

    def create_session(self):
        """
        Creates an empty session inside user-selected ws_dir.
        """
        # proceed only if user selected and accepted a ws_dir
        ws_dir = QtWidgets.QFileDialog.getExistingDirectory(
            parent=self.main_window, caption="Create new session",
            dir=self.workspace_dir)
        if not ws_dir:
            return
        # If sess already open, ask if delete (and return if backed out)
        can_continue = self.delete_existing_session_if_exists()
        if not can_continue:
            return
        # create and populate a session folder inside ws_dir
        sess_name = self.new_session_name()
        sess_dir = os.path.join(ws_dir, sess_name)
        h5w_path, h5m_path, h5o_path = self.process_session_dir(
            sess_dir, err_if_missing=False)
        os.mkdir(sess_dir)
        # create new HDF5 databases
        h5w_chunksize = self.h5_chunk_numhops * self.logmel_fn.hopsize
        h5w = SessionHDF5(h5w_path, height=1,
                          dtype=DemoSession.NP_DTYPE,
                          data_chunk_length=h5w_chunksize,
                          metadata_chunk_length=1, from_scratch=True)
        h5m = SessionHDF5(h5m_path, height=self.num_mels,
                          dtype=DemoSession.NP_DTYPE,
                          data_chunk_length=self.h5_chunk_numhops,
                          metadata_chunk_length=1, from_scratch=True)
        h5o = SessionHDF5(h5o_path, height=self.num_piano_keys,
                          dtype=DemoSession.NP_DTYPE,
                          data_chunk_length=self.h5_chunk_numhops,
                          metadata_chunk_length=1, from_scratch=True)
        # create empty session
        try:
            self.session = QtDemoSession(
                self.logmel_fn, self.ov_model, h5w, h5m, h5o,
                self.wav_samplerate, self.audio_recording_numhops,
                self.main_window.analysis_pan.get_detector_params,
                main_app=self)
            self.update_frontend_from_session()
        except Exception as e:
            h5w.close()
            h5m.close()
            h5o.close()
            raise RuntimeError(e)
        # If we reached this point, we were able to load the new session
        self.workspace_dir = os.path.dirname(sess_dir)

    def load_session(self):
        """
        """
        # proceed only if user selected and accepted a sess_dir
        sess_dir = QtWidgets.QFileDialog.getExistingDirectory(
            parent=self.main_window, caption="Load existing session",
            dir=self.workspace_dir)
        if not sess_dir:
            return
        # Check files in sess_path exist (raises an error otherwise)
        h5w_path, h5m_path, h5o_path = self.process_session_dir(
            sess_dir, err_if_missing=True)
        # If sess already open, ask if delete (and return if backed out)
        can_continue = self.delete_existing_session_if_exists()
        if not can_continue:
            return

        # At this point, user selected+confirmed a valid path, and old session
        # was terminated and removed. Load new session
        h5w_chunksize = self.h5_chunk_numhops * self.logmel_fn.hopsize
        h5w = SessionHDF5(h5w_path, height=1,
                          dtype=DemoSession.NP_DTYPE,
                          data_chunk_length=h5w_chunksize,
                          metadata_chunk_length=1, from_scratch=False)
        h5m = SessionHDF5(h5m_path, height=self.num_mels,
                          dtype=DemoSession.NP_DTYPE,
                          data_chunk_length=self.h5_chunk_numhops,
                          metadata_chunk_length=1, from_scratch=False)
        h5o = SessionHDF5(h5o_path, height=self.num_piano_keys,
                          dtype=DemoSession.NP_DTYPE,
                          data_chunk_length=self.h5_chunk_numhops,
                          metadata_chunk_length=1, from_scratch=False)
        try:
            self.session = QtDemoSession(
                self.logmel_fn, self.ov_model, h5w, h5m, h5o,
                self.wav_samplerate, self.audio_recording_numhops,
                self.main_window.analysis_pan.get_detector_params,
                main_app=self)
            self.update_frontend_from_session()  # possibly not needed?
        except Exception as e:
            h5w.close()
            h5m.close()
            h5o.close()
            raise RuntimeError(e)
        # If we reached this point, we were able to load the new session
        self.workspace_dir = os.path.dirname(sess_dir)

    def load_audio_file(self):
        """
        """
        # proceed only if session is present
        if self.session is None:
            self.nosess_dialog = NoSessionDialog(self.INFO_DIALOG_TIMEOUT_MS)
            self.nosess_dialog.show()
            return
        # proceed only if session is not recording:
        if self.session.is_recording():
            self.rec_dialog = RecDialog(self.INFO_DIALOG_TIMEOUT_MS)
            self.rec_dialog.show()
            return
        # proceed only if user selects and accepts a file
        audiopath, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent=self.main_window, caption="Load audio file",
            dir=self.load_audio_dir, filter=self.LOAD_AUDIO_FILTER)
        if not audiopath:
            return
        #
        self.session.add_wav_file(audiopath, normalize=True)
        self.update_frontend_from_session()
        self.load_audio_dir = os.path.dirname(audiopath)

    def record_audio(self):
        """
        """
        # proceed only if session is present
        if self.session is None:
            self.nosess_dialog = NoSessionDialog(self.INFO_DIALOG_TIMEOUT_MS)
            self.nosess_dialog.show()
            return
        if self.session.is_recording():
            self.session.stop_recording()
            self.update_frontend_from_session()
            self.main_window.analysis_pan.set_rec_but(recording=False)
        else:
            self.session.start_recording()
            self.main_window.analysis_pan.set_rec_but(recording=True)

    @QtCore.Slot()
    def update_frontend_from_session(self, new_xpos="maintain"):
        """
        To be called after changes in the session
        """
        if self.session is None:
            return
        # update mel and roll
        try:

            self.main_window.specroll.update(
                self.session.h5m.get_data(), self.session.h5o.get_data(),
                new_xpos=new_xpos)
        except ValueError:  # if session arrays empty, no spec update needed
            pass
        # update player
        player = self.main_window.analysis_pan.player
        arr = self.session.h5w.get_data()[0]
        arr_duration = float(len(arr)) / self.wav_samplerate
        player.set_array(arr, self.wav_samplerate)
        player.configure_gui(min_val=0, max_val=arr_duration,
                             step=0.1, pos_secs=0,
                             enable_buttons=True)

    @QtCore.Slot()
    def update_analysis_selrect(self):
        """
        to be called whenever a selection is issued in the piano roll
        """
        roll_view = self.main_window.specroll.roll_view
        analface = self.main_window.analysis_pan.analface
        #
        selnotes = roll_view.selected_notes
        vgrid = roll_view.get_vgrid_positions()
        #
        # if notes were selected:
        if selnotes:
            # update velocity histogram
            xxx, yyy, vvv = zip(*selnotes)
            analface.update_hist(
                vvv, nbins=self.num_analysis_histbins, rng=(0, 1))
            # compute velocity stats
            median_v = np.median(vvv)
            mean_v = np.mean(vvv)
            stats = [("Mean intensity", mean_v), ("Median intensity", median_v)]
            # if a vgrid was selected, add grid-error stats
            if vgrid is not None:
                vgrid_arr = np.array((vgrid[0], *vgrid[1], *vgrid[2]))
                min_diffs = self.grid_matching(xxx, vgrid_arr)
                #
                grid_stddev = np.mean([diff ** 2 for diff in min_diffs]) ** 0.5
                grid_median = np.median(min_diffs)
                stats.extend([("Grid error stddev", grid_stddev),
                              ("Grid error median", grid_median)])
            #
            analface.update_stats(stats)


    @QtCore.Slot()
    def update_analysis_vgrid(self):
        """
        to be called whenever a vertical grid is set in the piano roll.
        It updates the stats report
        """
        print(NotImplemented)


    @staticmethod
    def grid_matching(candidates, targets):
        """
        Given 2 scalar np arrays, it finds the closest target for each
        candidate, and gives the ``(t - c)``  difference for each candidate.
        """
        result = []
        for cand in candidates:
            diffs = targets - cand
            closest_idx = abs(diffs).argmin()
            closest_diff = diffs[closest_idx]
            result.append(closest_diff)
        #
        return result



# ##############################################################################
# # CLI
# ##############################################################################
@dataclass
class ConfDef:
    """
    :cvar INFERENCE_CHUNK_SIZE: In seconds, how big are chunks during inference
    :cvar INFERENCE_CHUNK_OVERLAP: In s, overlap between consecutive chunks
    :cvar OVIF: Onset-Value inference frames. Tuple of ``(lpad, body, rpad)``.
      See ``DemoSession`` for details.
    :cvar ZOOM_PERCENTAGE: When zooming in/out, the width will
      decrease/increase by this percentage.
    """
    APP: str = "iamusica_demo"
    DISPLAY_WIDTH: int = 1000
    AUDIO_RECORDING_NUMHOPS: int = 4
    PTHRESH: float = 0.5
    OVIF: List[int] = (10, 50, 10)
    MEL_OFFSET: float = 0.0
    MEL_VSHIFT: int = 0
    WORKSPACE_DIR: Optional[str] = None
    DARK_MODE: bool = True
    SPEC_CMAP: str = "bone"
    ROLL_CMAP: str = "cubehelix"
    VGRID_WIDTH: float = 10
    ZOOM_PERCENTAGE: float = 5
    NUM_ANALYSIS_BINS: int = 50


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
def run_iamusica_demo_app(initial_display_width=400,
                          audio_recording_numhops=4,
                          initial_pthresh=0.5,
                          initial_ovif=(10, 50, 10),
                          initial_mel_offset=0,
                          initial_mel_vshift=0,
                          initial_spec_cmap="bone",
                          initial_roll_cmap="cubehelix",
                          initial_vgrid_width=10,
                          zoom_percentage=5,
                          num_analysis_histbins=50,
                          workspace_dir=None,
                          dark_mode=True):
    """
    """

    app = IAMusicaApp("IAMUSICA DEMO", initial_display_width,
                      WAV_SAMPLERATE, MEL_FRAME_SIZE, MEL_FRAME_HOP, NUM_MELS,
                      MEL_FMIN, MEL_FMAX, MEL_WINDOW, OV_MODEL_PATH,
                      OV_MODEL_CONV1X1_HEAD, OV_MODEL_LRELU_SLOPE,
                      NUM_PIANO_KEYS,
                      #
                      audio_recording_numhops,
                      initial_ovif[1] + 10,  # h5 chunks larger than body
                      initial_pthresh,
                      initial_ovif,
                      initial_mel_offset,
                      initial_mel_vshift,
                      initial_spec_cmap,
                      initial_roll_cmap,
                      initial_vgrid_width,
                      zoom_percentage,
                      num_analysis_histbins,
                      workspace_dir)
    if dark_mode:
        app.setStyleSheet(qdarkstyle.load_stylesheet())
    app.main_window.show()
    # Wrap any exceptions into a dialog
    sys.excepthook = ExceptionDialog.excepthook
    # run app
    sys.exit(app.exec_())


if __name__ == '__main__':
    CONF = OmegaConf.structured(ConfDef())
    cli_conf = OmegaConf.from_cli()
    CONF = OmegaConf.merge(CONF, cli_conf)
    print("\n\nCONFIGURATION:")
    print(OmegaConf.to_yaml(CONF), end="\n\n\n")

    #
    if CONF.APP == "iamusica_demo":
        run_iamusica_demo_app(CONF.DISPLAY_WIDTH,
                              CONF.AUDIO_RECORDING_NUMHOPS,
                              CONF.PTHRESH, CONF.OVIF,
                              CONF.MEL_OFFSET, CONF.MEL_VSHIFT,
                              CONF.SPEC_CMAP,
                              CONF.ROLL_CMAP,
                              CONF.VGRID_WIDTH,
                              CONF.ZOOM_PERCENTAGE,
                              CONF.NUM_ANALYSIS_BINS,
                              CONF.WORKSPACE_DIR,
                              CONF.DARK_MODE)
    else:
        print("Unknown app name!")
