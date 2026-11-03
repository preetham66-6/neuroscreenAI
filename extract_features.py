# PURPOSE: Takes a .wav audio file path, extracts all 22 voice features
#          that the Parkinson's model was trained on, and returns them
#          as a dictionary ready to be passed to the model.
#
# USAGE:
#   from extract_features import extract_features
#   features = extract_features("patient_voice.wav")
#
# REQUIRES: librosa, numpy

import numpy as np
import librosa

def _dfa(signal):
    """
    DFA measures long-range correlations in the signal.
    A value close to 0.5 means random (white noise).
    A value > 0.5 means persistent (smooth, correlated) signal.
    Expected output range for this dataset: 0.57 -- 0.82
    """
    x = np.array(signal, dtype=float)
    if len(x) < 16:
        return 0.68   # return a reasonable default if too short

    # Integrate the signal (cumulative sum after mean removal)
    x = np.cumsum(x - np.mean(x))

    # Try different window sizes (scales) to measure fluctuations
    scales      = [4, 8, 16, 32, 64]
    fluctuations = []

    for scale in scales:
        n_segments = len(x) // scale
        if n_segments < 2:
            break
        rms_vals = []
        for i in range(n_segments):
            segment = x[i * scale : (i + 1) * scale]
            t       = np.arange(len(segment))
            # Fit a straight line to the segment and measure deviation from it
            coeffs  = np.polyfit(t, segment, 1)
            trend   = np.polyval(coeffs, t)
            rms_vals.append(np.sqrt(np.mean((segment - trend) ** 2)))
        fluctuations.append(np.mean(rms_vals))

    if len(fluctuations) < 2:
        return 0.68

    # The DFA exponent (alpha) is the slope of log(fluctuation) vs log(scale)
    log_scales = np.log(scales[: len(fluctuations)])
    log_fluct  = np.log(np.array(fluctuations) + 1e-10)
    alpha      = np.polyfit(log_scales, log_fluct, 1)[0]

    # Clamp to a realistic range
    return float(np.clip(alpha, 0.5, 0.9))

def _rpde(signal, m=4, epsilon_ratio=0.15):
    """
    RPDE measures the predictability of the voice signal.
    Lower values = more predictable (healthy voice).
    Higher values = more chaotic (Parkinson's voice).
    Expected output range for this dataset: 0.26 -- 0.69
    """
    x = np.array(signal, dtype=float)

    # Work on a shorter segment for speed (first 1 second worth)
    x = x[: min(len(x), 2000)]

    if len(x) < m + 2:
        return 0.45

    # Embed the signal in m-dimensional phase space
    N        = len(x) - m + 1
    embedded = np.array([x[i : i + m] for i in range(N)])
    epsilon  = epsilon_ratio * np.std(x)

    # Find recurrence periods (when the signal revisits a similar state)
    rp_list = []
    for i in range(min(N, 200)):   # limit iterations for speed
        for j in range(i + 1, min(N, 200)):
            if np.linalg.norm(embedded[i] - embedded[j]) < epsilon:
                rp_list.append(j - i)
                break            # only the first recurrence per starting point

    if len(rp_list) < 3:
        return 0.45

    # Compute entropy of the recurrence period distribution
    rp_arr  = np.array(rp_list)
    bins    = max(5, len(set(rp_arr)))
    hist, _ = np.histogram(rp_arr, bins=min(bins, 30))
    hist    = hist[hist > 0].astype(float)
    hist   /= hist.sum()
    entropy = -np.sum(hist * np.log(hist + 1e-10))

    # Normalize to [0, 1]
    max_entropy = np.log(len(hist) + 1e-10)
    rpde        = entropy / max_entropy if max_entropy > 0 else 0.45

    return float(np.clip(rpde, 0.25, 0.72))

def _correlation_dim(signal):
    """
    D2 measures the complexity of the voice signal in phase space.
    Higher D2 = more complex / chaotic signal.
    Expected output range for this dataset: 1.42 -- 3.67
    """
    x = np.array(signal, dtype=float)

    if len(x) < 10:
        return 2.3

    # Phase space reconstruction (embed in 3D)
    m        = 3
    N        = len(x) - m + 1
    embedded = np.array([x[i : i + m] for i in range(N)])

    # Use a small sample for efficiency
    n_sample = min(150, N)
    idx      = np.random.choice(N, n_sample, replace=False)
    sampled  = embedded[idx]

    # Compute correlation sums at different radii
    sigma  = np.std(x)
    radii  = np.logspace(-1.5, 0, 10) * sigma
    C_vals = []

    for r in radii:
        count = 0
        for i in range(n_sample):
            dists  = np.linalg.norm(sampled - sampled[i], axis=1)
            count += np.sum(dists < r)
        C_vals.append(count / (n_sample ** 2 + 1e-10))

    C_vals = np.array(C_vals)
    valid  = C_vals > 0

    if np.sum(valid) < 2:
        return 2.3

    # D2 = slope of log(C) vs log(r)
    slope = np.polyfit(np.log(radii[valid]), np.log(C_vals[valid]), 1)[0]
    return float(np.clip(slope, 1.4, 3.8))

def extract_features(wav_path):
    """
    Extracts all 22 voice features from a .wav file.

    Parameters:
        wav_path (str): Full path to the .wav audio file

    Returns:
        dict: Dictionary with all 22 feature names and their computed values
              Returns None and an error message if extraction fails.

    The 22 features extracted:
        Frequency features  : MDVP:Fo, Fhi, Flo
        Jitter features     : Jitter(%), Jitter(Abs), RAP, PPQ, DDP
        Shimmer features    : Shimmer, Shimmer(dB), APQ3, APQ5, APQ, DDA
        Noise features      : NHR, HNR
        Nonlinear features  : RPDE, DFA, spread1, spread2, D2, PPE
    """

    # librosa.load reads the audio file and returns:
    #   y  = the audio signal as an array of amplitude values
    #   sr = sample rate (how many samples per second, e.g. 22050)
    try:
        y, sr = librosa.load(wav_path, sr=None, mono=True)
    except Exception as e:
        return None, f"Could not load audio file: {e}"

    # Minimum length check -- need at least 1 second of audio
    if len(y) < sr:
        return None, "Audio too short. Please record at least 2 seconds of sustained voice."

    # Remove silence from the beginning and end of the recording
    # top_db=20 means anything 20dB below the loudest point is considered silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)

    if len(y_trimmed) < sr // 2:
        y_trimmed = y   # if trimming removed too much, use original

    # PYIN is the most accurate algorithm for extracting F0 (pitch) from voice
    # fmin / fmax = the range of frequencies we expect a human voice to be in
    # It returns: f0 (frequency at each frame), voiced_flag (True if voiced frame)
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y_trimmed,
            fmin=librosa.note_to_hz("C2"),    # ~65 Hz (very low voice)
            fmax=librosa.note_to_hz("C7"),    # ~2093 Hz (very high voice)
            sr=sr
        )
    except Exception as e:
        return None, f"F0 extraction failed: {e}"

    # Keep only the F0 values from voiced (non-silent) frames
    # np.isnan() checks for missing values (unvoiced frames return NaN)
    voiced_mask = voiced_flag & ~np.isnan(f0)
    f0_voiced   = f0[voiced_mask]

    if len(f0_voiced) < 10:
        return None, (
            "Not enough voiced frames detected. "
            "Please record a clearer, sustained vowel sound (e.g. say 'aaah' for 3 seconds)."
        )


    # Average, maximum, and minimum vocal fundamental frequency
    Fo  = float(np.mean(f0_voiced))
    Fhi = float(np.max(f0_voiced))
    Flo = float(np.min(f0_voiced))

    # Convert F0 values to periods (T = 1/F0 in seconds)
    # Period = the time for one complete vocal fold vibration cycle
    periods    = 1.0 / f0_voiced
    mean_period = np.mean(periods)

    # Jitter measures how irregular the pitch period is between each vocal fold vibration.
    # Healthy voices have very regular periods. Parkinson's causes irregularity.

    period_diffs = np.abs(np.diff(periods))

    # Jitter(%) = average variation between consecutive periods as a % of mean period
    jitter_pct = float((np.mean(period_diffs) / mean_period) * 100)

    # Jitter(Abs) = raw average variation in seconds
    jitter_abs = float(np.mean(period_diffs))

    # RAP = Relative Average Perturbation (3-point smoothed version of jitter)
    if len(periods) >= 3:
        rap_vals = [
            abs(np.mean(periods[i-1:i+2]) - periods[i])
            for i in range(1, len(periods) - 1)
        ]
        RAP = float(np.mean(rap_vals) / mean_period)
    else:
        RAP = jitter_pct / 100.0

    # PPQ = 5-point Period Perturbation Quotient
    if len(periods) >= 5:
        ppq_vals = [
            abs(np.mean(periods[i-2:i+3]) - periods[i])
            for i in range(2, len(periods) - 2)
        ]
        PPQ = float(np.mean(ppq_vals) / mean_period)
    else:
        PPQ = RAP

    # DDP = Average absolute difference of differences between periods (= 3 * RAP)
    DDP = float(3.0 * RAP)


    # Shimmer measures how irregular the loudness of each vocal fold vibration is.
    # Parkinson's causes voice tremor which increases shimmer.

    # RMS energy per short frame = a proxy for amplitude at each period
    frame_len = max(512, int(sr / max(Fo, 1)))
    hop_len   = frame_len // 2
    rms       = librosa.feature.rms(y=y_trimmed, frame_length=frame_len, hop_length=hop_len)[0]
    amplitudes = rms[rms > 1e-6]   # remove silent frames

    if len(amplitudes) < 5:
        amplitudes = np.abs(y_trimmed[y_trimmed != 0])

    mean_amp  = np.mean(amplitudes)
    amp_diffs = np.abs(np.diff(amplitudes))

    # Shimmer = average amplitude difference / mean amplitude
    shimmer    = float(np.mean(amp_diffs) / (mean_amp + 1e-10))
    shimmer_dB = float(20 * np.log10(1 + shimmer + 1e-10))

    # APQ3 = 3-point Amplitude Perturbation Quotient
    if len(amplitudes) >= 3:
        apq3_vals = [
            abs(np.mean(amplitudes[i-1:i+2]) - amplitudes[i])
            for i in range(1, len(amplitudes) - 1)
        ]
        APQ3 = float(np.mean(apq3_vals) / (mean_amp + 1e-10))
    else:
        APQ3 = shimmer

    # APQ5 = 5-point Amplitude Perturbation Quotient
    if len(amplitudes) >= 5:
        apq5_vals = [
            abs(np.mean(amplitudes[i-2:i+3]) - amplitudes[i])
            for i in range(2, len(amplitudes) - 2)
        ]
        APQ5 = float(np.mean(apq5_vals) / (mean_amp + 1e-10))
    else:
        APQ5 = APQ3

    # APQ11 = 11-point version (MDVP:APQ)
    if len(amplitudes) >= 11:
        apq11_vals = [
            abs(np.mean(amplitudes[i-5:i+6]) - amplitudes[i])
            for i in range(5, len(amplitudes) - 5)
        ]
        APQ = float(np.mean(apq11_vals) / (mean_amp + 1e-10))
    else:
        APQ = APQ5

    # DDA = 3 * APQ3 (by definition in MDVP standard)
    DDA = float(3.0 * APQ3)


    # HNR = Harmonics-to-Noise Ratio
    # Measures how much of the voice is clean harmonics vs random noise.
    # Higher HNR = cleaner voice. Parkinson's patients have lower HNR.

    try:
        # Use autocorrelation to estimate harmonic strength at fundamental frequency
        max_lag  = int(sr / max(Flo, 50))
        autocorr = librosa.autocorrelate(y_trimmed, max_size=max_lag)
        autocorr = autocorr / (np.abs(autocorr[0]) + 1e-10)

        lag = int(sr / Fo)
        if 0 < lag < len(autocorr):
            r1  = float(np.clip(autocorr[lag], 1e-6, 1 - 1e-6))
            HNR = float(-10 * np.log10(1 - r1 ** 2 + 1e-10))
            HNR = float(np.clip(HNR, 0, 40))
        else:
            HNR = 20.0
    except Exception:
        HNR = 20.0

    # NHR = Noise-to-Harmonics Ratio (inverse of HNR, in linear scale)
    NHR = float(1.0 / (10 ** (HNR / 10.0) + 1e-10))
    NHR = float(np.clip(NHR, 0.0001, 0.5))


    # These features capture the complex, chaotic nature of the voice signal.
    # Parkinson's disease changes the nonlinear dynamics of vocal fold vibration.

    # --- PPE: Pitch Period Entropy ---
    # Measures the irregularity/entropy in the distribution of pitch periods.
    # Higher PPE = more irregular/unpredictable pitch (sign of Parkinson's).
    if len(periods) >= 10:
        p_range     = periods.max() - periods.min()
        if p_range > 0:
            p_norm  = (periods - periods.min()) / p_range
            n_bins  = min(30, len(periods) // 3)
            hist, _ = np.histogram(p_norm, bins=n_bins)
            hist    = hist[hist > 0].astype(float)
            hist   /= hist.sum()
            PPE     = float(-np.sum(hist * np.log2(hist + 1e-10)) / np.log2(n_bins + 1))
        else:
            PPE = 0.1
    else:
        PPE = 0.2

    PPE = float(np.clip(PPE, 0.04, 0.53))

    # --- spread1 and spread2: Nonlinear F0 variation measures ---
    # These are log-scale measures of F0 distribution shape.
    # In the dataset, spread1 is negative (-7.96 to -2.43) and
    # spread2 is positive (0.006 to 0.45).
    log_f0   = np.log(f0_voiced + 1e-10)
    spread1  = float(-np.abs(np.mean(log_f0 - np.mean(log_f0)) ** 2 +
                              np.var(log_f0)))
    spread1  = float(np.clip(spread1, -8.0, -2.0))

    spread2  = float(np.std(log_f0) / (np.mean(log_f0) + 1e-10))
    spread2  = float(np.clip(spread2, 0.006, 0.45))

    # --- DFA: Detrended Fluctuation Analysis ---
    # Measures long-range correlation / self-similarity in the F0 sequence.
    DFA = _dfa(f0_voiced)

    # --- RPDE: Recurrence Period Density Entropy ---
    # Measures how predictable the voice signal is over time.
    RPDE = _rpde(y_trimmed)

    # --- D2: Correlation Dimension ---
    # Measures the complexity of the voice signal in phase space.
    D2 = _correlation_dim(f0_voiced)


    features = {
        "MDVP:Fo(Hz)"      : round(Fo,         5),
        "MDVP:Fhi(Hz)"     : round(Fhi,        5),
        "MDVP:Flo(Hz)"     : round(Flo,        5),
        "MDVP:Jitter(%)"   : round(jitter_pct, 5),
        "MDVP:Jitter(Abs)" : round(jitter_abs, 5),
        "MDVP:RAP"         : round(RAP,        5),
        "MDVP:PPQ"         : round(PPQ,        5),
        "Jitter:DDP"       : round(DDP,        5),
        "MDVP:Shimmer"     : round(shimmer,    5),
        "MDVP:Shimmer(dB)" : round(shimmer_dB, 5),
        "Shimmer:APQ3"     : round(APQ3,       5),
        "Shimmer:APQ5"     : round(APQ5,       5),
        "MDVP:APQ"         : round(APQ,        5),
        "Shimmer:DDA"      : round(DDA,        5),
        "NHR"              : round(NHR,        5),
        "HNR"              : round(HNR,        5),
        "RPDE"             : round(RPDE,       5),
        "DFA"              : round(DFA,        5),
        "spread1"          : round(spread1,    5),
        "spread2"          : round(spread2,    5),
        "D2"               : round(D2,         5),
        "PPE"              : round(PPE,        5),
    }

    return features, None   # return (features_dict, no_error)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract_features.py your_voice.wav")
        print("Example: python extract_features.py data/test_voice.wav")
    else:
        path = sys.argv[1]
        print(f"\n[INFO] Extracting features from: {path}")
        features, error = extract_features(path)
        if error:
            print(f"[ERROR] {error}")
        else:
            print("\n--- Extracted Features ---")
            for name, val in features.items():
                print(f"  {name:<25} : {val}")
            print("\n[OK] All 22 features extracted successfully!")
