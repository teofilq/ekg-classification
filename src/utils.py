import numpy as np
import neurokit2 as nk


class ECGAnalyzer:
    """
    Example class for extracting ECG metrics from a single-lead ECG of length N.
    Uses NeuroKit2 internally to find peaks, delineate waves, and derive durations.

    Parameters
    ----------
    ecg_signal : array-like
        The (already filtered) ECG signal (or second derivative, but see disclaimer) of length N.
    sampling_rate : int
        Sampling frequency in Hz. Default is 500.
    method_peaks : str
        The R-peak detection method to use (e.g., 'neurokit', 'pantompkins1985', etc.).
    method_delineate : str
        The wave delineation method to use (e.g., 'dwt', 'peak', 'prominence', ...).
    """

    def __init__(
        self,
        ecg_signal,
        sampling_rate=500,
        method_peaks="neurokit",
        method_delineate="dwt",
    ):
        self.ecg_signal = np.array(ecg_signal)
        self.sr = sampling_rate
        self.method_peaks = method_peaks
        self.method_delineate = method_delineate

        # --- 1) Detect R-peaks ---
        # Because your signal is already filtered (?), you can pass it directly to ecg_peaks.
        # If your second derivative is not recognized, you may get poor results.
        # Feel free to add correct_artifacts=True if needed.
        signals_r, info_r = nk.ecg_peaks(self.ecg_signal,
                                         sampling_rate=self.sr,
                                         method=self.method_peaks)

        # Store R-peak indices
        self.rpeaks_idx = info_r["ECG_R_Peaks"]  # list of sample indices where R-peaks occur

        # --- 2) QRS Count ---
        self.qrs_count = len(self.rpeaks_idx)

        # --- 3) Compute Ventricular Rate (from R-peaks) ---
        # By default, it returns the instantaneous rate at the location of each R-peak.
        # Optionally, you can interpolate over the entire signal length by specifying desired_length.
        self.ventricular_rate = nk.ecg_rate(self.rpeaks_idx,
                                            sampling_rate=self.sr,
                                            desired_length=None)  # or len(ecg_signal)

        # --- 4) Delineate Waves (to get Q, T, P info) ---
        # This will find the onsets/offsets/peaks of QRS, T, P waves.
        # If your second derivative truly does not look like an ECG, this might fail.
        self.waves_signals, self.waves_info = nk.ecg_delineate(
            ecg_cleaned=self.ecg_signal,
            rpeaks=self.rpeaks_idx,
            sampling_rate=self.sr,
            method=self.method_delineate
        )

    def get_ventricular_rate(self):
        """
        Returns the instantaneous (per-beat) Ventricular Rate from R-peaks.
        Units are in beats per minute (bpm).
        """
        return self.ventricular_rate

    def get_atrial_rate(self):
        """
        Attempts to compute the Atrial Rate via P-peaks.
        If P-peaks weren't detected or aren't present, returns None or an empty array.
        """
        if "ECG_P_Peaks" not in self.waves_info:
            return None
        ppeaks_idx = self.waves_info["ECG_P_Peaks"]
        if len(ppeaks_idx) < 2:
            return None

        # Convert P-peaks to instantaneous rate exactly like R-peaks
        atrial_rate = nk.ecg_rate(ppeaks_idx,
                                  sampling_rate=self.sr,
                                  desired_length=None)
        return atrial_rate

    def get_qrs_duration(self):
        """
        Returns a list/array of QRS durations (in seconds) for each detected beat.
        By default, NeuroKit2 provides R-onsets and R-offsets for the QRS complex.
        Duration = R_offset - R_onset (in samples), then converted to seconds.
        """
        # Check that we have the right keys from delineation
        if "ECG_R_Onsets" not in self.waves_info or "ECG_R_Offsets" not in self.waves_info:
            return None

        onsets = self.waves_info["ECG_R_Onsets"]
        offsets = self.waves_info["ECG_R_Offsets"]

        # QRS durations for each beat
        durations = []
        for onset, offset in zip(onsets, offsets):
            dur_samples = offset - onset
            dur_sec = dur_samples / self.sr
            durations.append(dur_sec)
        return durations

    def get_qt_intervals(self):
        """
        Returns a list of QT intervals (in seconds) for each beat.
        By definition, QT interval = QRS onset to T-offset.
        The wave delineation in NeuroKit2 typically calls QRS onset 'ECG_R_Onsets'.
        T-offset is in 'ECG_T_Offsets'.
        """
        if ("ECG_R_Onsets" not in self.waves_info or 
            "ECG_T_Offsets" not in self.waves_info):
            return None

        q_onsets = self.waves_info["ECG_R_Onsets"]
        t_offsets = self.waves_info["ECG_T_Offsets"]

        # We must match each Q-onset to the subsequent T-offset
        # Typically, the nth QRS onset lines up with the nth T offset for the same cycle,
        # but always check array lengths in real data.
        qt_list = []
        for q_idx, t_idx in zip(q_onsets, t_offsets):
            interval_samples = t_idx - q_idx
            interval_sec = interval_samples / self.sr
            qt_list.append(interval_sec)
        return qt_list

    def get_qrs_count(self):
        """
        Simply the number of R-peaks detected.
        """
        return self.qrs_count

    def get_q_onset_offsets(self):
        """
        NeuroKit2 doesn't explicitly label 'Q_onset' or 'Q_offset' columns.
        Typically, Q wave is within the QRS complex, whose boundaries are R_Onsets, R_Offsets.
        If you truly need a Q wave's boundaries, you'll have to interpret them from Q-peaks
        or from morphological definitions. As a placeholder, you could do:

        Q_onset = R_onset
        Q_offset ~ Q-peak or local minima to R-peak, etc.

        This is a rough approximation. Here, we just return Q-peaks if found.
        """
        if "ECG_Q_Peaks" not in self.waves_info:
            return None, None

        # Q peaks array
        q_peaks = self.waves_info["ECG_Q_Peaks"]

        # In standard ECG nomenclature, Q onset is basically R onset,
        # so we do a naive version: map each Q peak to the nearest R onset.
        # This is not truly "Q onset," but a placeholder for demonstration.

        if "ECG_R_Onsets" in self.waves_info:
            q_onsets = self.waves_info["ECG_R_Onsets"]
        else:
            q_onsets = None

        # There's no direct "Q_Offsets" in NK. You might approximate it by local minima
        # or times halfway to R-peak, etc. Here we’ll just return None as an example.
        q_offsets = None

        return q_onsets, q_offsets


# # -----------------------------------------------------------------------
# # Example usage
# if __name__ == "__main__":

#     # Example: create a synthetic ECG (10s, 500Hz => 5000 samples)
#     # If you already have your own 2nd-derivative array, replace 'ecg_signal' below with it.
#     ecg_signal = nk.ecg_simulate(duration=10, sampling_rate=500, heart_rate=70)

#     analyzer = ECGAnalyzer(ecg_signal, sampling_rate=500)

#     print("QRS Count:", analyzer.get_qrs_count())
#     print("Ventricular Rate (bpm) [per-beat]:", analyzer.get_ventricular_rate())
#     print("QRS Duration (s) [per-beat]:", analyzer.get_qrs_duration())
#     print("QT Intervals (s) [per-beat]:", analyzer.get_qt_intervals())

#     # Attempt at Atrial Rate detection (P-peaks)
#     atrial_rate = analyzer.get_atrial_rate()
#     if atrial_rate is None:
#         print("No P-peaks detected or not enough data for Atrial Rate.")
#     else:
#         print("Atrial Rate (bpm) [per-beat]:", atrial_rate)

#     # Q wave onset/offset placeholders
#     q_onset, q_offset = analyzer.get_q_onset_offsets()
#     print("Q onset indices:", q_onset)
#     print("Q offset indices:", q_offset)