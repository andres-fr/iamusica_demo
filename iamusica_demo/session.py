#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
"""


# from threading import Thread
#
import numpy as np
import h5py
import torch
import torch.nn.functional as F
#
from .audio_loop import AsynchAudioInputStream
from .utils import torch_load_resample_audio


# ##############################################################################
# # HDF5
# ##############################################################################
class SessionHDF5:
    """
    Edit matrices of same height and arbitrary width. Note the usage of very
    few datasets, to prevent slow loading times.
    """
    DATA_NAME = "data"
    METADATA_NAME = "metadata"
    IDXS_NAME = "data_idxs"

    def __init__(self, out_path, height, dtype=np.float32, compression="lzf",
                 data_chunk_length=500, metadata_chunk_length=500,
                 from_scratch=False):
        """
        :param height: The height of the HDF5 stored matrix is fixed to this.
        :param compression: ``lzf`` is fast, ``gzip`` slower but provides
          better compression
        :param data_chunk_length: Every I/O operation goes by chunks. A too
          small chunk size will cause many syscalls (slow), and with a too
          large chunk size we will be loading too much information in a single
          syscall (also slow, and bloats the RAM). Ideally, the chunk length is
          a bit larger than what is usually needed (e.g. if we expect to read
          between 10 and 50 rows at a time, we can choose chunk=60).
        :param mode: File handle protocol. ``r+`` stands for R+W, file must
          exist. With ``a``, file is created if doesn't exist.
        :param from_scratch: If true, file is assumed to not exist beforehand.
          Otherwise, file is assumed to exist.
        """
        self.out_path = out_path
        self.height = height
        self.dtype = dtype
        self.compression = compression
        #
        if from_scratch:
            self.h5f = h5py.File(out_path, "x")
            self.data_ds = self.h5f.create_dataset(
                self.DATA_NAME, shape=(height, 0), maxshape=(height, None),
                dtype=dtype, compression=compression,
                chunks=(height, data_chunk_length))
            self.metadata_ds = self.h5f.create_dataset(
                self.METADATA_NAME, shape=(0,), maxshape=(None,),
                compression=compression, dtype=h5py.string_dtype(),
                chunks=(metadata_chunk_length,))
            self.data_idxs_ds = self.h5f.create_dataset(
                self.IDXS_NAME, shape=(2, 0), maxshape=(2, None),
                dtype=np.int64, compression=compression,
                chunks=(2, metadata_chunk_length))
        else:
            self.h5f = h5py.File(out_path, "r+")
            self.data_ds = self.h5f[self.DATA_NAME]
            self.metadata_ds = self.h5f[self.METADATA_NAME]
            self.data_idxs_ds = self.h5f[self.IDXS_NAME]

    def __enter__(self):
        """
        """
        return self

    def __exit__(self, type, value, traceback):
        """
        """
        self.close()

    def close(self):
        """
        """
        self.h5f.close()

    def append(self, matrix, metadata_str):
        """
        :param matrix: dtype array of shape ``(fix_height, width)``
        """
        n = self.get_num_elements(self.h5f)
        current_width = self.get_data_shape(self.h5f)[1]
        # n = self._num_entries
        h, w = matrix.shape
        assert h == self.height, \
            f"Shape was {(h, w)} but should be ({self.height}, ...). "
        # update arr size and add data
        new_data_w = current_width + w
        self.data_ds.resize((self.height, new_data_w))
        self.data_ds[:, current_width:new_data_w] = matrix
        # # update meta-arr size and add metadata
        self.metadata_ds.resize((n + 1,))
        self.metadata_ds[n] = metadata_str
        # update data-idx size and add entry
        self.data_idxs_ds.resize((2, n + 1))
        self.data_idxs_ds[:, n] = (current_width, new_data_w)
        #
        self.h5f.flush()

    def get_data(self, beg=None, end=None):
        """
        """
        result = self.h5f[SessionHDF5.DATA_NAME][:, beg:end]
        return result

    @classmethod
    def get_element(cls, h5file, elt_idx):
        """
        :param int elt_idx: Index of the appended element, e.g. first element
          has index 0, second has index 1...
        :returns: the ``(data, metadata_str)`` corresponding to that index,
          as they were appended.
        """
        data_beg, data_end = h5file[cls.IDXS_NAME][:, elt_idx]
        data = h5file[cls.DATA_NAME][:, data_beg:data_end]
        metadata = h5file[cls.METADATA_NAME][elt_idx].decode("utf-8")
        return data, metadata

    @classmethod
    def get_num_elements(cls, h5file):
        """
        :returns: The number of elements that have been added to the file via
          append.
        """
        num_elements = len(h5file[cls.METADATA_NAME])
        return num_elements

    @classmethod
    def get_data_shape(cls, h5file):
        """
        :returns: Shape of the full data matrix
        """
        result = h5file[cls.DATA_NAME].shape
        return result


# ##############################################################################
# # SESSION
# ##############################################################################
class DemoSession:
    """
    """
    NP_DTYPE = np.float32
    FILL_MEL = "min"
    FILL_ROLL = 0

    def save_session(self, h5w_chunk_numhops=4, h5mel_chunk_numframes=50):
        """
        """
        raise NotImplementedError("DemoSession.save")

    def __init__(self, logmel_fn, ov_model, h5w, h5m, h5o,
                 wav_samplerate=16000, wav_in_numhops=4,
                 inference_params_fn=lambda: (0.5, (10, 50, 10), 0, 0)):
        """
        :param initial_ovif: See ``check_ovif``.
        :param inference_params_fn: A parameterless function that retrieves 4
          elements needed for the inference:
          * probability threshold: Between 0 and 1, detections below this will
            be ignored.
          * ovif: is short for Onset+Velocity inference frames, and is a triple
            of non-negative integers ``(left_trim, body, right_trim)``,
            determining the size of the inference window: the sum of the 3 is
            passed to the model, and the sides are trimmed after inference.
          * mel gain: Float to be added to the mel spectrogram
          * mel vshift: Real-valued index designing the vertical shift to be
            applied to the mel spectrogram. A vshfit of x means that
            ``spec[i]`` will be moved to ``spec[i+x]``. Empty margins will be
            filled with the min value for that spectrogram.
        """

        # self.inf_thread = None  # ATM thread not used
        #
        # self.ts = make_timestamp()
        # sess_dir = os.path.join(workspace_dir, self.ts)
        # os.mkdir(sess_dir)
        # h5wav_path = os.path.join(sess_dir, "wav_" + self.ts + ".h5")
        # h5mel_path = os.path.join(sess_dir, "mel_" + self.ts + ".h5")
        # h5onset_path = os.path.join(sess_dir, "onset_" + self.ts + ".h5")

        # pointers to the inference moduoles
        assert logmel_fn.samplerate == wav_samplerate, \
            "Mismatching samplerates!"
        self.logmel_fn = logmel_fn
        self.ov_model = ov_model
        self._inference_params_fn = inference_params_fn
        #
        self.h5w, self.h5m, self.h5o = h5w, h5m, h5o
        # audio input stream
        self.in_chunk_length = logmel_fn.hopsize * wav_in_numhops
        self.aais = AsynchAudioInputStream(
            samplerate=wav_samplerate, chunk_length=self.in_chunk_length,
            update_callback=self.wav_update_callback)

    def get_inference_params(self):
        """
        """
        pthresh, ovif, offs, vshift = self._inference_params_fn()
        trim_l, body, trim_r = ovif
        # check ovif
        assert all(x >= 0 for x in ovif), \
            "ov_inference_frames entries must be nonnegative!"
        assert body > 0, "OVIF body must be strictly positive!"
        assert trim_l <= (body + trim_r), \
            "left_trim can't be bigger than body+right_trim!"
        # check rest
        assert 0 <= pthresh <= 1, "Pthresh must be a probability!"
        assert np.issubdtype(np.array(vshift).dtype, np.integer), \
            "Vshift must be an integer!"
        #
        return (pthresh, ovif, offs, vshift)

    # CONVENIENCE WRAPPERS
    def check_h5len_consistency(self):
        """
        """
        len_w, len_m, len_o = self.get_h5_lengths()
        assert len_m == len_o, "Insconsistent len_m and len_o!"
        assert len_w == (len_m * self.logmel_fn.hopsize), \
            "Inconsistent len_w and len_m!"

    def is_recording(self):
        """
        """
        result = self.aais.stream.is_active()
        return result

    def get_h5_lengths(self):
        """
        :returns: Current lengths of the wav, mel and onset HDF5 files, in that
          order.
        """
        len_w = self.h5w.h5f[SessionHDF5.DATA_NAME].shape[1]
        len_m = self.h5m.h5f[SessionHDF5.DATA_NAME].shape[1]
        len_o = self.h5o.h5f[SessionHDF5.DATA_NAME].shape[1]
        return (len_w, len_m, len_o)

    def fill_blank_mel_roll(self, num_frames, fill_mel="min", fill_roll=0):
        """
        Fills end of mel and roll HDF5 files for the given number of frames,
        with the given values. Useful e.g. to compensate for recording latency.
        """
        roll_arr = np.full((self.h5o.height, num_frames),
                           fill_value=fill_roll, dtype=self.NP_DTYPE)
        self.h5o.append(roll_arr, metadata_str="")
        # figure out fill_mel value
        if fill_mel == "min":
            fill_mel = self.logmel_fn(
                torch.zeros(self.logmel_fn.winsize)).min()
        #
        mel_arr = np.full((self.h5m.height, num_frames),
                          fill_value=fill_mel, dtype=self.NP_DTYPE)
        self.h5m.append(mel_arr, metadata_str="")

    def fill_blank_mel_roll_up_to_wav(self, fill_mel="min", fill_roll=0):
        """
        if last gathered wav chunks were insufficient for inference,
        fill up remaining log+roll, to ensure wav/log/roll consistency.
        We could do inference for smaller chunks, but we treat the chunksize
        as a requirement we can't infer below.
        """
        len_w, len_m, _ = self.get_h5_lengths()
        w_blocks = (len_w / self.logmel_fn.hopsize)
        assert w_blocks.is_integer(), \
            "Length of wav array is not a multiple of hop size?"
        diff_chunks = int(w_blocks) - len_m
        self.fill_blank_mel_roll(diff_chunks, fill_mel, fill_roll)
        # state must be consistent when stopping recording
        self.check_h5len_consistency()

    # CONTEXT/STATE HANDLING
    def __enter__(self):
        """
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        """
        self.terminate()

    def start_recording(self):
        """
        """
        if not self.is_recording():
            # State must be consistent when starting recording
            self.check_h5len_consistency()
            # If we have left trimming, fill the corresponding mel+prediction
            # at beginning with empty values, to ensure consistency
            _, (trim_l, _, _), _, _ = self.get_inference_params()
            if trim_l > 0:
                self.fill_blank_mel_roll(trim_l, self.FILL_MEL, self.FILL_ROLL)
            #
            self.aais.start()
            print("[SESSION] started. Recording...")

    def stop_recording(self):
        """
        """
        if self.is_recording():
            self.aais.stop()
            self.fill_blank_mel_roll_up_to_wav(self.FILL_MEL, self.FILL_ROLL)
            print("[SESSION] stopped")

    def terminate(self):
        """
        Close the input stream (can't record afterwards) and close h5 handles.
        """
        if self.is_recording():
            self.stop_recording()
        self.aais.terminate()
        self.h5w.close()
        self.h5m.close()
        self.h5o.close()
        print("[SESSION] terminated")

    # LIVE RECORDING+PREDICTIONS
    @staticmethod
    def predict_wav(logmel_fn, ov_model, wav, pthresh=0.5, mel_offset=0,
                    mel_vshift=0, trim_out_l=0, trim_out_r=0):
        """
        :param wav: Torch tensor of shape ``(L,)`` to be fed to ``logmel_fn``.
        :param trim_out_l: Nonnegative. Given output of shape ``(k, t)``,
          remove the first l indexes. Also remove corresponding entries in
          dataframe.
        :param trim_out_r: Nonnegative. Given output of shape ``(k, t)``,
          remove the last r indexes. Also remove corresponding entries in
          dataframe.
        :returns: ``(trimmed_logmel, trimmed_roll, trimmed_df)``.
       """
        print("\n\n PREDICTING WITH:", pthresh, (mel_offset, mel_vshift),
              (trim_out_l, trim_out_r), wav.shape)
        logmel = logmel_fn(wav, mel_offset, mel_vshift)
        ov_roll, df = ov_model(logmel, pthresh)
        # optionally trim
        if trim_out_l > 0:
            logmel = logmel[:, trim_out_l:]
            ov_roll = ov_roll[:, trim_out_l:]
            df = df[df["t_idx"] >= trim_out_l]
            df["t_idx"] -= trim_out_l
        if trim_out_r > 0:
            logmel = logmel[:, :-trim_out_r]
            ov_roll = ov_roll[:, :-trim_out_r]
            df = df[df["t_idx"] < logmel.shape[1]]
        return logmel, ov_roll, df

    def predict_and_update(self, wav_beg, wav_end):
        """
        This convenience method wraps ``predict_wav``. It is part of the
        ``wav_update_callback`` logic (being called with appropiate ``beg`` and
        ``end`` indexes whenever there is enough freshly recorded wav to
        perform inference), but written separate so it can also be called
        asynchronously if needed.
        It computes the corresponding logmel and onset+velocity maps, and adds
        them to the corresponding HDF5 files.
        """
        thresh, (trim_l, _, trim_r), offs, vshift = self.get_inference_params()
        #
        wav_chunk = torch.from_numpy(
            self.h5w.get_data(wav_beg, wav_end)[0])
        mel, roll, df = self.predict_wav(
            self.logmel_fn, self.ov_model, wav_chunk,
            thresh, offs, vshift, trim_l, trim_r)
        print(df)
        del df  # dataframe ignored ATM
        # add trimmed inference results to corresponding h5 files
        self.h5m.append(mel, metadata_str="")
        self.h5o.append(roll, metadata_str="")

    def wav_update_callback(self, chunk):
        """
        This callback function gets called by the audio collection loop.
        Needed to perform real-time inference while recording.
        :returns: True if inference was performed after adding this wav
          chunk, false otherwise.
        """
        _, (trim_l, body, trim_r), _, _ = self.get_inference_params()
        # add audio chunk to h5, and update h5 lengths
        self.h5w.append(chunk[None, :], metadata_str="")
        # If current wav length is ahead of mel_len by more than
        # (body+right_padding) frames, we can extend our predictions from
        # (mel_len-left_padding) to (mel_len + body + right_padding).
        # Since we are trimming the padding, this will effectively extend
        # only by the body length
        h5w_len, h5m_len, _ = self.get_h5_lengths()
        mel_len_samples = self.logmel_fn.hopsize * h5m_len
        body_right_samples = self.logmel_fn.hopsize * (body + trim_r)
        can_extend = ((h5w_len - mel_len_samples) >= body_right_samples)
        if can_extend:
            wav_beg = mel_len_samples - (self.logmel_fn.hopsize * trim_l)
            wav_end = mel_len_samples + body_right_samples
            self.predict_and_update(wav_beg, wav_end)
            return True
        else:
            # if can't extend, just keep adding audio until there is enough
            return False

    # FILE LOADING+PREDICTIONS
    def add_wav_file(self, wav_path, normalize=True):
        """
        """
        print(f"[SESSION] adding wav file: {wav_path}")
        # load wav into 1D array and ensure it is a multiple of hopsize
        wav = torch_load_resample_audio(
            wav_path, target_sr=self.logmel_fn.samplerate, mono=True,
            normalize=True, device="cpu")
        _, rest = divmod(len(wav), self.logmel_fn.hopsize)
        if rest != 0:
            wav = F.pad(wav, (0, self.logmel_fn.hopsize - rest))
            assert divmod(len(wav), self.logmel_fn.hopsize)[1] == 0, \
                "Wav still not multiple of hopsize?"
            trim_r = 1
        else:
            trim_r = 0
        #
        thresh, (trim_l, _, trim_r), offs, vshift = self.get_inference_params()
        mel, roll, df = self.predict_wav(
            self.logmel_fn, self.ov_model, wav,
            thresh, offs, vshift, trim_l, trim_r)
        print(df)
        del df  # dataframe ignored ATM
        # add trimmed inference results to corresponding h5 files
        self.h5w.append(wav.unsqueeze(0), metadata_str="")
        self.h5m.append(mel, metadata_str="")
        self.h5o.append(roll, metadata_str="")
        self.fill_blank_mel_roll_up_to_wav(self.FILL_MEL, self.FILL_ROLL)
        print(f"[SESSION] added! ({mel.shape}, {roll.shape})")
