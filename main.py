import argparse
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

try:
    import sounddevice as sd
except Exception:
    sd = None

from matplotlib.ticker import ScalarFormatter

A4_MIDI = 69
A4_FREQ = 440.0
PIANO_START = 21
PIANO_END = 108

NOTE_NAMES_SHARP = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]

NOTE_ALIASES = {
    "DB": "C#",
    "EB": "D#",
    "GB": "F#",
    "AB": "G#",
    "BB": "A#",
}

# -----------------------------------------------------------------------------
# Style d'accord "balanced-like"
# -----------------------------------------------------------------------------
# Important : ces poids ne sont pas "mesurés" par l'audio.
# Ils DEFINISSENT la fonction objectif que l'optimiseur minimise.
# Sans cible externe, chercher automatiquement les poids serait mal posé :
# la solution triviale serait de mettre tous les poids à zéro.
#
# Ici, le style donne beaucoup de poids aux octaves et aux douzièmes.
# Le poids 1.00 est affiché comme 100 % sur les graphes.
# -----------------------------------------------------------------------------

BALANCED_LIKE_INTERVALS = [
    # family, step_semitones, partial_low, partial_high, weight
    # Octaves : plusieurs manières d'entendre une octave pure.
    ("octave", 12, 2, 1, 0.80),
    ("octave", 12, 4, 2, 1.00),
    ("octave", 12, 6, 3, 0.75),
    ("octave", 12, 8, 4, 0.45),
    ("octave", 12, 10, 5, 0.25),

    # Douzièmes : ratio acoustique 3:1, très important dans le style équilibré.
    ("douzieme", 19, 3, 1, 1.00),
    ("douzieme", 19, 6, 2, 0.65),
    ("douzieme", 19, 9, 3, 0.35),

    # Intervalles secondaires : stabilisent la solution sans prendre le dessus.
    ("quinte", 7, 3, 2, 0.20),
    ("quinte", 7, 6, 4, 0.10),
    ("quarte", 5, 4, 3, 0.10),
    ("double_octave", 24, 4, 1, 0.30),
    ("double_octave", 24, 8, 2, 0.18),
    ("dixneuvieme", 31, 6, 1, 0.20),
]

FAMILY_LABELS = {
    "octave": "Octaves",
    "douzieme": "Douzièmes",
    "quinte": "Quintes",
    "quarte": "Quartes",
    "double_octave": "Doubles octaves",
    "dixneuvieme": "Dix-neuvièmes",
}


# -----------------------------------------------------------------------------
# Notes, fréquences, cents
# -----------------------------------------------------------------------------

def midi_to_freq(midi: int, a4: float = A4_FREQ) -> float:
    return float(a4 * 2 ** ((midi - A4_MIDI) / 12))


def midi_to_name(midi: int) -> str:
    return f"{NOTE_NAMES_SHARP[midi % 12]}{midi // 12 - 1}"


def parse_note_name(token: str) -> int:
    t = token.strip().upper().replace("♯", "#").replace("♭", "B")

    if not t:
        raise ValueError("Nom de note vide")

    letter = t[0]
    rest = t[1:]

    acc = ""
    if rest.startswith("#") or rest.startswith("B"):
        acc = rest[0]
        rest = rest[1:]

    name = letter + acc
    name = NOTE_ALIASES.get(name, name)
    octave = int(rest)

    return NOTE_NAMES_SHARP.index(name) + 12 * (octave + 1)


def cents(freq: float, ref: float) -> float:
    return 1200.0 * math.log2(freq / ref)


# -----------------------------------------------------------------------------
# Analyse audio note par note
# -----------------------------------------------------------------------------

def quadratic_interpolated_peak(freqs, mag, idx):
    if idx <= 0 or idx >= len(mag) - 1:
        return float(freqs[idx]), float(mag[idx])

    y0, y1, y2 = np.log(np.maximum(mag[idx - 1:idx + 2], 1e-30))
    denom = y0 - 2 * y1 + y2

    if abs(denom) < 1e-18:
        return float(freqs[idx]), float(mag[idx])

    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -0.5, 0.5))

    bin_hz = freqs[1] - freqs[0]
    peak_freq = freqs[idx] + delta * bin_hz
    peak_mag = math.exp(y1 - 0.25 * (y0 - y2) * delta)

    return float(peak_freq), float(peak_mag)


def analyse_note_audio(
    audio,
    fs,
    midi,
    max_partials=10,
    trim_attack_s=0.15,
    search_cents=140.0,
    min_rel_amp=0.025,
):
    """
    Analyse une note enregistrée.

    Retourne :
    - un DataFrame des partiels détectés ;
    - un résumé : f0, B, écart en cents, qualité du fit.

    Le coefficient B est estimé à partir de :
        f_k = k * f0 * sqrt(1 + B * k^2)
    """

    x = np.asarray(audio, dtype=float).reshape(-1)
    start = min(int(trim_attack_s * fs), max(0, len(x) - 1))
    x = x[start:]

    if len(x) < fs // 4:
        raise ValueError("Signal trop court")

    x = x - np.mean(x)
    peak = np.max(np.abs(x))

    if peak <= 1e-9:
        raise ValueError("Signal trop faible ou muet")

    x = x / peak

    win = np.hanning(len(x))
    nfft = 1 << int(np.ceil(np.log2(len(x) * 4)))

    spec = np.fft.rfft(x * win, n=nfft)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(nfft, 1 / fs)

    f_et = midi_to_freq(midi)
    rows = []
    global_max = float(np.max(mag))

    for k in range(1, max_partials + 1):
        pred = k * f_et

        if pred >= 0.92 * fs / 2:
            break

        lo = pred * 2 ** (-search_cents / 1200)
        hi = pred * 2 ** (search_cents / 1200)
        idxs = np.flatnonzero((freqs >= lo) & (freqs <= hi))

        if len(idxs) < 3:
            continue

        i_peak = int(idxs[np.argmax(mag[idxs])])
        f_peak, a_peak = quadratic_interpolated_peak(freqs, mag, i_peak)
        rel_amp = a_peak / global_max if global_max > 0 else 0.0

        if rel_amp >= min_rel_amp:
            rows.append({
                "midi": midi,
                "note": midi_to_name(midi),
                "partial": k,
                "freq_hz": f_peak,
                "rel_amp": rel_amp,
                "search_center_hz": pred,
            })

    partials = pd.DataFrame(rows)

    if len(partials) < 2:
        summary = {
            "midi": midi,
            "note": midi_to_name(midi),
            "f0_hz": np.nan,
            "B": np.nan,
            "measured_cents_vs_et": np.nan,
            "fit_rms_cents": np.nan,
            "n_partials": len(partials),
        }
        return partials, summary

    k = partials["partial"].to_numpy(float)
    f_obs = partials["freq_hz"].to_numpy(float)
    w = partials["rel_amp"].to_numpy(float)
    w = np.sqrt(w / max(np.max(w), 1e-12))

    def residual(params):
        f0, B = params
        pred = k * f0 * np.sqrt(1.0 + B * k * k)
        return w * 1200.0 * np.log2(np.maximum(pred, 1e-12) / f_obs)

    x0 = np.array([f_et, 1e-4])
    lower = np.array([f_et * 2 ** (-200 / 1200), 0.0])
    upper = np.array([f_et * 2 ** (200 / 1200), 2e-2])

    sol = least_squares(
        residual,
        x0=x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=3.0,
    )

    f0, B = sol.x
    rms = float(np.sqrt(np.mean(residual(sol.x) ** 2)))

    summary = {
        "midi": midi,
        "note": midi_to_name(midi),
        "f0_hz": float(f0),
        "B": float(B),
        "measured_cents_vs_et": cents(float(f0), f_et),
        "fit_rms_cents": rms,
        "n_partials": int(len(partials)),
    }

    return partials, summary


# -----------------------------------------------------------------------------
# Enregistrement
# -----------------------------------------------------------------------------

def record_audio(seconds, fs, device=None):
    if sd is None:
        raise RuntimeError(
            "Le module sounddevice n'est pas installé ou PortAudio est indisponible."
        )

    rec = sd.rec(
        int(seconds * fs),
        samplerate=fs,
        channels=1,
        dtype="float64",
        device=device,
    )
    sd.wait()
    return rec[:, 0]


def measure_live(midis, seconds, fs, outdir, device=None):
    outdir.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_partials = []

    print("\nMode enregistrement note par note.")
    print("Conseil : isole une seule corde, joue mezzo-forte, puis laisse sonner.\n")

    for midi in midis:
        note = midi_to_name(midi)
        input(f"Prépare {note}. Appuie sur Entrée puis joue immédiatement la note...")

        print("Enregistrement...")
        audio = record_audio(seconds, fs, device=device)

        wav_path = outdir / f"audio_{midi:03d}_{note.replace('#', 's')}.npy"
        np.save(wav_path, audio)

        try:
            partials, summary = analyse_note_audio(audio, fs, midi)
            print(
                f"  {note}: "
                f"f0={summary['f0_hz']:.3f} Hz, "
                f"B={summary['B']:.3e}, "
                f"écart ET={summary['measured_cents_vs_et']:.2f} c, "
                f"partiels={summary['n_partials']}"
            )
        except Exception as exc:
            print(f"  Analyse impossible pour {note}: {exc}")
            partials = pd.DataFrame()
            summary = {
                "midi": midi,
                "note": note,
                "f0_hz": np.nan,
                "B": np.nan,
                "measured_cents_vs_et": np.nan,
                "fit_rms_cents": np.nan,
                "n_partials": 0,
            }

        summaries.append(summary)

        if len(partials):
            all_partials.append(partials)

        pd.DataFrame(summaries).to_csv(outdir / "measurements.csv", index=False)

        if all_partials:
            pd.concat(all_partials, ignore_index=True).to_csv(
                outdir / "partials.csv",
                index=False,
            )

    measurements = pd.DataFrame(summaries)
    partials = pd.concat(all_partials, ignore_index=True) if all_partials else pd.DataFrame()
    return measurements, partials


def upsert_single_note(outdir, summary, partials):
    """
    Remplace une note existante dans measurements.csv et partials.csv.
    Si la note n'existe pas encore, elle est ajoutée.
    """

    outdir.mkdir(parents=True, exist_ok=True)

    midi = int(summary["midi"])
    measurements_path = outdir / "measurements.csv"
    partials_path = outdir / "partials.csv"
    new_summary = pd.DataFrame([summary])

    if measurements_path.exists():
        old = pd.read_csv(measurements_path)
        old = old[old["midi"] != midi]
        measurements = pd.concat([old, new_summary], ignore_index=True)
    else:
        measurements = new_summary

    measurements = measurements.sort_values("midi").reset_index(drop=True)
    measurements.to_csv(measurements_path, index=False)

    if partials_path.exists():
        old_partials = pd.read_csv(partials_path)
        old_partials = old_partials[old_partials["midi"] != midi]
    else:
        old_partials = pd.DataFrame()

    if partials is not None and len(partials):
        all_partials = pd.concat([old_partials, partials], ignore_index=True)
    else:
        all_partials = old_partials

    if len(all_partials):
        all_partials = all_partials.sort_values(["midi", "partial"]).reset_index(drop=True)
        all_partials.to_csv(partials_path, index=False)

    return measurements


def measure_single_note(note_name, seconds, fs, outdir, device=None):
    """
    Enregistre une seule note, remplace ses anciennes données,
    puis s'arrête sans recalculer les graphes.
    """

    midi = parse_note_name(note_name)
    note = midi_to_name(midi)

    print(f"\nMode note seule : {note}")
    input(f"Prépare {note}. Appuie sur Entrée puis joue immédiatement la note...")

    print("Enregistrement...")
    audio = record_audio(seconds, fs, device=device)

    outdir.mkdir(parents=True, exist_ok=True)
    audio_path = outdir / f"audio_{midi:03d}_{note.replace('#', 's')}.npy"
    np.save(audio_path, audio)

    partials, summary = analyse_note_audio(audio, fs, midi)
    upsert_single_note(outdir, summary, partials)

    print(
        f"\nNote remplacée : {note}\n"
        f"  f0 = {summary['f0_hz']:.3f} Hz\n"
        f"  B = {summary['B']:.3e}\n"
        f"  écart tempérament égal = {summary['measured_cents_vs_et']:.2f} cents\n"
        f"  partiels détectés = {summary['n_partials']}\n"
    )
    print("Les graphes n'ont pas été régénérés.")


# -----------------------------------------------------------------------------
# Courbe d'inharmonicité lisse, sans courbe théorique
# -----------------------------------------------------------------------------

def fit_smooth_inharmonicity_curve(measurements, midis, default_B=1e-4):
    """
    Ajuste une courbe lisse idéale de B.

    Ce n'est PAS une courbe théorique.
    C'est une courbe orange lisse, ajustée sur les points mesurés.

    Modèle utilisé :
        log10(B) = a + b*z + c*z^2
    avec z = position MIDI normalisée autour de A4.

    La contrainte c >= 0 donne une forme globalement convexe, proche d'une
    forme en x^2 sur l'axe des notes.
    """

    midis = np.asarray(midis, dtype=float)
    z_all = (midis - A4_MIDI) / 39.0

    fit_info = {
        "fit_type": "fallback_default",
        "a": np.nan,
        "b": np.nan,
        "c": np.nan,
        "n_points": 0,
        "rms_log10": np.nan,
    }

    if measurements is None or measurements.empty or "B" not in measurements.columns:
        return np.full(len(midis), float(default_B)), fit_info

    m = measurements["midi"].to_numpy(float)
    b_meas = measurements["B"].to_numpy(float)
    valid = np.isfinite(m) & np.isfinite(b_meas) & (b_meas > 0)

    if np.sum(valid) == 0:
        return np.full(len(midis), float(default_B)), fit_info

    m_valid = m[valid]
    b_valid = b_meas[valid]
    z_valid = (m_valid - A4_MIDI) / 39.0
    y_valid = np.log10(b_valid)

    fit_info["n_points"] = int(len(y_valid))

    if len(y_valid) == 1:
        fit_info["fit_type"] = "single_measured_B"
        return np.full(len(midis), float(b_valid[0])), fit_info

    if len(y_valid) == 2:
        fit_info["fit_type"] = "log_linear_interpolation"
        y_all = np.interp(midis, m_valid, y_valid, left=y_valid[0], right=y_valid[-1])
        return 10 ** np.clip(y_all, -8, -1), fit_info

    # Pondération optionnelle : plus il y a de partiels et plus le fit local est bon,
    # plus le point compte. Si les colonnes n'existent pas, tout vaut 1.
    if "n_partials" in measurements.columns:
        n_partials = measurements.loc[valid, "n_partials"].to_numpy(float)
        w_partials = np.sqrt(np.clip(n_partials, 1, None) / np.nanmax(np.clip(n_partials, 1, None)))
    else:
        w_partials = np.ones_like(y_valid)

    if "fit_rms_cents" in measurements.columns:
        rms = measurements.loc[valid, "fit_rms_cents"].to_numpy(float)
        rms = np.where(np.isfinite(rms), rms, np.nanmedian(rms[np.isfinite(rms)]) if np.any(np.isfinite(rms)) else 3.0)
        w_rms = 1.0 / np.sqrt(1.0 + np.clip(rms, 0, 50) / 3.0)
    else:
        w_rms = np.ones_like(y_valid)

    weights = w_partials * w_rms

    # Initialisation par polyfit sans contrainte, puis projection de c vers positif.
    p = np.polyfit(z_valid, y_valid, deg=2, w=np.maximum(weights, 1e-6))
    # np.polyfit renvoie c2, c1, c0.
    x0 = np.array([p[2], p[1], max(p[0], 1e-6)], dtype=float)

    def residual(params):
        a, b, c = params
        y_pred = a + b * z_valid + c * z_valid * z_valid
        return weights * (y_pred - y_valid)

    sol = least_squares(
        residual,
        x0=x0,
        bounds=([-8.0, -10.0, 0.0], [-1.0, 10.0, 10.0]),
        loss="soft_l1",
        f_scale=0.20,
        max_nfev=3000,
    )

    a, b, c = sol.x
    y_all = a + b * z_all + c * z_all * z_all
    B_all = 10 ** np.clip(y_all, -8, -1)

    rms_log10 = float(np.sqrt(np.mean((a + b * z_valid + c * z_valid * z_valid - y_valid) ** 2)))

    fit_info.update({
        "fit_type": "smooth_quadratic_log10_B",
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "rms_log10": rms_log10,
    })

    return B_all, fit_info


# -----------------------------------------------------------------------------
# Partiels, deviations, tuning curve balanced-like
# -----------------------------------------------------------------------------

def partial_frequency(F, B, partial):
    return partial * F * np.sqrt(1.0 + B * partial * partial)


def compute_interval_deviation(
    F_low,
    B_low,
    F_high,
    B_high,
    p_low,
    p_high,
):
    """
    Déviation d'un couple de partiels.

    high_partial - low_partial en Hz :
        > 0 : le partiel de la note haute est au-dessus du partiel de la note basse.
        < 0 : le partiel de la note haute est en-dessous.

    deviation_cents :
        1200 * log2(high_partial / low_partial)

    Zéro = coïncidence parfaite des partiels = intervalle pur pour ce couple.
    """

    low_partial = partial_frequency(F_low, B_low, p_low)
    high_partial = partial_frequency(F_high, B_high, p_high)

    beat_hz = high_partial - low_partial
    deviation_cents = 1200.0 * np.log2(np.maximum(high_partial, 1e-12) / np.maximum(low_partial, 1e-12))

    return float(deviation_cents), float(beat_hz), float(low_partial), float(high_partial)


def optimize_tuning_curve(
    measurements,
    intervals=BALANCED_LIKE_INTERVALS,
    midi_start=PIANO_START,
    midi_end=PIANO_END,
    a4=A4_FREQ,
    smooth_weight=0.08,
    anchor_weight=12.0,
    max_abs_offset_cents=120.0,
):
    """
    Calcule une tuning curve "balanced-like".

    Variables optimisées : offsets en cents de chaque note par rapport au tempérament égal.

    Objectif minimisé :
        Σ poids_intervalle * deviation_intervalle_cents²
        + poids_lissage * courbure_de_la_tuning_curve²
        + poids_ancrage * offset_A4²

    Les poids des intervalles ne sont pas trouvés par cette optimisation :
    ils sont les réglages du style d'accord.
    """

    midis = np.arange(midi_start, midi_end + 1)
    n = len(midis)
    et = np.array([midi_to_freq(int(m), a4=a4) for m in midis])

    B_used, B_fit_info = fit_smooth_inharmonicity_curve(measurements, midis)
    anchor_idx = int(np.argmin(np.abs(midis - A4_MIDI)))

    def residual(offsets_cents):
        F = et * 2 ** (offsets_cents / 1200.0)
        res = []

        for family, step, p_low, p_high, weight in intervals:
            if weight <= 0:
                continue

            for i in range(n - step):
                j = i + step
                dev_cents, _, _, _ = compute_interval_deviation(
                    F[i], B_used[i], F[j], B_used[j], p_low, p_high
                )
                res.append(math.sqrt(weight) * dev_cents)

        if n >= 3 and smooth_weight > 0:
            res.extend(list(math.sqrt(smooth_weight) * np.diff(offsets_cents, n=2)))

        res.append(math.sqrt(anchor_weight) * offsets_cents[anchor_idx])
        return np.asarray(res, dtype=float)

    sol = least_squares(
        residual,
        x0=np.zeros(n),
        bounds=(-max_abs_offset_cents, max_abs_offset_cents),
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=6000,
    )

    offsets = sol.x
    target_hz = et * 2 ** (offsets / 1200.0)

    tuning = pd.DataFrame({
        "midi": midis,
        "note": [midi_to_name(int(m)) for m in midis],
        "et_hz": et,
        "B_used": B_used,
        "target_hz": target_hz,
        "target_cents_vs_et": offsets,
    })

    if measurements is not None and not measurements.empty:
        cols = [c for c in [
            "midi", "f0_hz", "B", "measured_cents_vs_et", "fit_rms_cents", "n_partials"
        ] if c in measurements.columns]
        measured = measurements[cols].copy()
        if "B" in measured.columns:
            measured = measured.rename(columns={"B": "B_measured"})
    else:
        measured = pd.DataFrame(columns=[
            "midi", "f0_hz", "B_measured", "measured_cents_vs_et", "fit_rms_cents", "n_partials"
        ])

    tuning = tuning.merge(measured, on="midi", how="left")

    if "measured_cents_vs_et" in tuning.columns:
        tuning["deviation_measured_vs_target_cents"] = (
            tuning["measured_cents_vs_et"] - tuning["target_cents_vs_et"]
        )
    else:
        tuning["deviation_measured_vs_target_cents"] = np.nan

    if "f0_hz" in tuning.columns:
        actual_hz = tuning["f0_hz"].to_numpy(float)
    else:
        actual_hz = np.full(n, np.nan)

    if "B_measured" in tuning.columns:
        B_measured = tuning["B_measured"].to_numpy(float)
        B_actual = np.where(np.isfinite(B_measured) & (B_measured > 0), B_measured, B_used)
    else:
        B_actual = B_used.copy()

    dev_rows = []

    for family, step, p_low, p_high, weight in intervals:
        for i in range(n - step):
            j = i + step
            dev_cents, beat_hz, low_partial_hz, high_partial_hz = compute_interval_deviation(
                target_hz[i], B_used[i], target_hz[j], B_used[j], p_low, p_high
            )

            if (
                np.isfinite(actual_hz[i])
                and np.isfinite(actual_hz[j])
                and actual_hz[i] > 0
                and actual_hz[j] > 0
            ):
                actual_dev_cents, actual_beat_hz, actual_low_partial_hz, actual_high_partial_hz = compute_interval_deviation(
                    actual_hz[i], B_actual[i], actual_hz[j], B_actual[j], p_low, p_high
                )
                actual_freq_low_hz = float(actual_hz[i])
                actual_freq_high_hz = float(actual_hz[j])
            else:
                actual_dev_cents = np.nan
                actual_beat_hz = np.nan
                actual_low_partial_hz = np.nan
                actual_high_partial_hz = np.nan
                actual_freq_low_hz = np.nan
                actual_freq_high_hz = np.nan

            dev_rows.append({
                "midi_low": int(midis[i]),
                "note_low": midi_to_name(int(midis[i])),
                "midi_high": int(midis[j]),
                "note_high": midi_to_name(int(midis[j])),
                "freq_low_hz": float(target_hz[i]),
                "freq_high_hz": float(target_hz[j]),
                "actual_freq_low_hz": actual_freq_low_hz,
                "actual_freq_high_hz": actual_freq_high_hz,
                "interval_family": family,
                "interval_label": FAMILY_LABELS.get(family, family),
                "step_semitones": int(step),
                "partial_low": int(p_low),
                "partial_high": int(p_high),
                "partial_ratio": f"{p_low}:{p_high}",
                "weight": float(weight),
                "weight_percent": float(100.0 * weight),
                "deviation_cents": float(dev_cents),
                "beat_hz": float(beat_hz),
                "beat_abs_hz": float(abs(beat_hz)),
                "low_partial_hz": float(low_partial_hz),
                "high_partial_hz": float(high_partial_hz),
                "actual_deviation_cents": float(actual_dev_cents),
                "actual_beat_hz": float(actual_beat_hz),
                "actual_beat_abs_hz": float(abs(actual_beat_hz)) if np.isfinite(actual_beat_hz) else np.nan,
                "actual_low_partial_hz": float(actual_low_partial_hz),
                "actual_high_partial_hz": float(actual_high_partial_hz),
            })

    interval_deviations = pd.DataFrame(dev_rows)
    return tuning, interval_deviations, B_fit_info


# -----------------------------------------------------------------------------
# Graphes
# -----------------------------------------------------------------------------

PUB_COLORS = {
    "orange": "#E69F00",
    "blue": "#0072B2",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#222222",
    "gray": "#666666",
    "lightgray": "#D9D9D9",
}


def configure_publication_style():
    """
    Configure matplotlib pour obtenir des graphes propres et prêts pour un article scientifique.
    N'utilise pas LaTeX externe : le rendu mathématique repose uniquement sur mathtext.
    """
    plt.rcParams.update({
        # Figure
        "figure.figsize": (6.8, 3.8),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,

        # Fonts
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,

        # Lines
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
        "axes.linewidth": 0.8,

        # Grid
        "grid.linewidth": 0.5,
        "grid.alpha": 0.22,

        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fancybox": False,
        "legend.edgecolor": "#CCCCCC",

        # Fonts / mathtext sans LaTeX externe
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "mathtext.fontset": "cm",

        # Tick directions
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,

        # Axes
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": False,
    })


def save_figure(fig, outdir, stem):
    """
    Sauvegarde en PNG.
    """
    png_path = outdir / f"{stem}.png"
    fig.savefig(png_path)
    return [png_path]


def style_axes(ax):
    """
    Style homogène des axes.
    """
    ax.grid(True, which="major", color=PUB_COLORS["lightgray"], alpha=0.35, linewidth=0.6)
    ax.grid(True, which="minor", color=PUB_COLORS["lightgray"], alpha=0.18, linewidth=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color(PUB_COLORS["black"])
    ax.tick_params(which="both", width=0.8, colors=PUB_COLORS["black"])


def setup_frequency_axis(ax):
    ax.set_xscale("log")

    ticks = [27.5, 55, 110, 220, 440, 880, 1760, 3520]
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xticklabels([f"{t:g}" for t in ticks])

    ax.set_xlabel(r"Fréquence de la note (Hz)")


def grouped_weight_text(intervals, max_lines=12):
    grouped = defaultdict(list)

    for family, step, p_low, p_high, weight in intervals:
        grouped[family].append(f"{p_low}:{p_high} {100 * weight:.0f}%")

    lines = ["Objectif minimisé : Σ w·déviation²", "Poids du style balanced-like :"]

    for family in ["octave", "douzieme", "quinte", "quarte", "double_octave", "dixneuvieme"]:
        if family in grouped:
            label = FAMILY_LABELS.get(family, family)
            lines.append(f"{label}: " + ", ".join(grouped[family]))

    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["…"]

    return "\n".join(lines)


def plot_inharmonicity(tuning, outdir):
    fig, ax = plt.subplots(figsize=(6.8, 3.9))

    ax.semilogy(
        tuning["et_hz"],
        tuning["B_used"],
        color=PUB_COLORS["orange"],
        linewidth=2.2,
        label=r"Courbe lisse ajustée",
        zorder=2,
    )

    if "B_measured" in tuning.columns:
        measured = tuning.dropna(subset=["B_measured"])
        if len(measured):
            x = measured["f0_hz"].where(measured["f0_hz"].notna(), measured["et_hz"])
            ax.scatter(
                x,
                measured["B_measured"],
                s=18,
                color=PUB_COLORS["black"],
                alpha=0.90,
                label=r"Points mesurés",
                zorder=3,
            )

    setup_frequency_axis(ax)
    ax.set_ylabel(r"Coefficient d'inharmonicité $B$")
    ax.set_title(r"Inharmonicité du piano")
    style_axes(ax)
    ax.legend(loc="best")

    fig.tight_layout()
    save_figure(fig, outdir, "01_inharmonicite_B")
    plt.close(fig)


def plot_tuning_curve(tuning, outdir, intervals):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    x = tuning["target_hz"].to_numpy(float)
    y = tuning["target_cents_vs_et"].to_numpy(float)

    # Courbe cible
    ax.plot(
        x,
        y,
        color=PUB_COLORS["orange"],
        linewidth=2.2,
        marker="o",
        markersize=3.2,
        markerfacecolor="white",
        markeredgewidth=0.8,
        label=r"Tuning curve calculée",
        zorder=2,
    )

    # Bandes ±10 cents
    ax.plot(
        x,
        y + 10.0,
        linestyle="--",
        linewidth=1.3,
        color=PUB_COLORS["blue"],
        label=r"$+10$ cents",
        zorder=1,
    )
    ax.plot(
        x,
        y - 10.0,
        linestyle="--",
        linewidth=1.3,
        color=PUB_COLORS["blue"],
        label=r"$-10$ cents",
        zorder=1,
    )

    # Points mesurés
    if {"f0_hz", "measured_cents_vs_et"}.issubset(tuning.columns):
        measured = tuning.dropna(subset=["f0_hz", "measured_cents_vs_et"])
        if len(measured):
            outliers = measured["measured_cents_vs_et"].abs() > 10.0
            normal = measured[~outliers]
            red = measured[outliers]

            if len(normal):
                ax.scatter(
                    normal["f0_hz"],
                    normal["measured_cents_vs_et"],
                    s=20,
                    color=PUB_COLORS["black"],
                    alpha=0.90,
                    label=r"Points mesurés",
                    zorder=3,
                )

            if len(red):
                ax.scatter(
                    red["f0_hz"],
                    red["measured_cents_vs_et"],
                    s=28,
                    color="#D62728",
                    alpha=0.95,
                    label=r"Points mesurés $|c| > 10$",
                    zorder=4,
                )

    ax.axhline(0, linewidth=1.0, color=PUB_COLORS["gray"], alpha=0.8)

    setup_frequency_axis(ax)
    ax.set_ylabel(r"Écart par rapport au tempérament égal (cents)")
    ax.set_title(r"Tuning curve de type balanced")
    style_axes(ax)

    ax.text(
        0.016,
        0.98,
        grouped_weight_text(intervals),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#BFBFBF",
            "alpha": 0.95,
        },
    )

    ax.legend(loc="lower right", ncol=1)
    fig.tight_layout()
    save_figure(fig, outdir, "02_tuning_curve_balanced_like")
    plt.close(fig)


def plot_deviation_curves(interval_deviations, outdir, deviation_unit="cents"):
    if interval_deviations.empty:
        return []

    if deviation_unit == "hz":
        y_col = "beat_hz"
        actual_y_col = "actual_beat_hz"
        ylabel = r"Écart entre partiels (Hz)"
        zero_note = r"$0$ Hz = coïncidence parfaite des partiels"
        stem_suffix = "hz"
    else:
        y_col = "deviation_cents"
        actual_y_col = "actual_deviation_cents"
        ylabel = r"Écart entre partiels (cents)"
        zero_note = r"$0$ cent = coïncidence parfaite des partiels"
        stem_suffix = "cents"

    files = []

    family_order = [
        "octave",
        "douzieme",
        "quinte",
        "quarte",
        "double_octave",
        "dixneuvieme",
    ]

    palette = [
        PUB_COLORS["blue"],
        PUB_COLORS["orange"],
        PUB_COLORS["green"],
        PUB_COLORS["vermillion"],
        PUB_COLORS["purple"],
        PUB_COLORS["sky"],
    ]

    for family in family_order:
        family_df = interval_deviations[interval_deviations["interval_family"] == family]
        if family_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(7.1, 4.3))

        for i, (ratio, grp) in enumerate(family_df.groupby("partial_ratio", sort=False)):
            weight = float(grp["weight"].iloc[0])
            color = palette[i % len(palette)]

            label = f"{ratio} — cible pondérée {100 * weight:.0f}%"
            ax.plot(
                grp["freq_low_hz"],
                grp[y_col],
                linewidth=1.8,
                color=color,
                label=label,
            )

            if {"actual_freq_low_hz", actual_y_col}.issubset(grp.columns):
                actual_grp = grp.dropna(subset=["actual_freq_low_hz", actual_y_col])
                if len(actual_grp):
                    actual_label = f"{ratio} — actuel"
                    ax.plot(
                        actual_grp["actual_freq_low_hz"],
                        actual_grp[actual_y_col],
                        linestyle="--",
                        linewidth=1.6,
                        color=color,
                        alpha=0.95,
                        label=actual_label,
                    )

        ax.axhline(0, linewidth=1.0, color=PUB_COLORS["gray"], alpha=0.8)

        setup_frequency_axis(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(rf"Deviation curve — {FAMILY_LABELS.get(family, family)}")
        style_axes(ax)

        ax.text(
            0.016,
            0.98,
            zero_note + "\n" + r"Signe : partiel haut $-$ partiel bas",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "#BFBFBF",
                "alpha": 0.95,
            },
        )

        ax.legend(loc="best", ncol=2, fontsize=8.0)
        fig.tight_layout()

        stem = f"03_deviation_{family}_{stem_suffix}"
        file_paths = save_figure(fig, outdir, stem)
        files.extend(file_paths)
        plt.close(fig)

    return files


def plot_results(tuning, interval_deviations, outdir, intervals, deviation_unit="cents"):
    outdir.mkdir(parents=True, exist_ok=True)

    configure_publication_style()

    plot_inharmonicity(tuning, outdir)
    plot_tuning_curve(tuning, outdir, intervals)
    return plot_deviation_curves(interval_deviations, outdir, deviation_unit=deviation_unit)


# -----------------------------------------------------------------------------
# Rapport d'analyse
# -----------------------------------------------------------------------------

def rms(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan
    return float(np.sqrt(np.mean(v * v)))


def markdown_table(rows, headers):
    def fmt(x):
        if isinstance(x, float):
            if not np.isfinite(x):
                return ""
            if abs(x) >= 100:
                return f"{x:.1f}"
            if abs(x) >= 10:
                return f"{x:.2f}"
            return f"{x:.3f}"
        return str(x)

    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


def write_analysis_report(tuning, interval_deviations, outdir, intervals, B_fit_info):
    lines = []
    lines.append("# Analyse des résultats\n")

    lines.append("## 1. Courbe d'inharmonicité\n")
    lines.append(
        "Le graphe `01_inharmonicite_B.png` ne contient pas de courbe théorique. "
        "Les points noirs sont les coefficients B mesurés note par note. "
        "La courbe orange est une courbe lisse ajustée sur ces points."
    )
    lines.append(
        "Le modèle utilisé est `log10(B) = a + b·z + c·z²`, avec `z` la position MIDI normalisée autour de A4. "
        "La contrainte `c >= 0` impose une forme globalement convexe, proche d'une forme en x²."
    )
    lines.append(
        f"Fit B : type = `{B_fit_info.get('fit_type')}`, "
        f"points = {B_fit_info.get('n_points')}, "
        f"a = {B_fit_info.get('a'):.6g}, "
        f"b = {B_fit_info.get('b'):.6g}, "
        f"c = {B_fit_info.get('c'):.6g}.\n"
    )

    lines.append("## 2. Tuning curve balanced-like\n")
    low = tuning.iloc[0]
    high = tuning.iloc[-1]
    min_row = tuning.loc[tuning["target_cents_vs_et"].idxmin()]
    max_row = tuning.loc[tuning["target_cents_vs_et"].idxmax()]

    lines.append(
        "Le graphe `02_tuning_curve_balanced_like.png` montre uniquement la courbe cible orange : "
        "c'est l'écart en cents entre la fréquence cible et le tempérament égal."
    )
    lines.append(
        "La courbe est calculée en minimisant une somme pondérée de déviations de partiels, "
        "avec un ancrage de A4 et un terme de lissage."
    )
    lines.append(
        "Point important : les poids ne sont pas découverts par l'optimisation. "
        "Ils sont les paramètres du style. L'optimisation trouve les fréquences cibles qui minimisent la somme pour ces poids."
    )
    lines.append(
        f"Étendue obtenue : {low['note']} = {low['target_cents_vs_et']:.2f} cents, "
        f"{high['note']} = {high['target_cents_vs_et']:.2f} cents. "
        f"Minimum : {min_row['note']} = {min_row['target_cents_vs_et']:.2f} cents. "
        f"Maximum : {max_row['note']} = {max_row['target_cents_vs_et']:.2f} cents.\n"
    )

    weight_rows = []
    for family, step, p_low, p_high, weight in intervals:
        weight_rows.append({
            "famille": FAMILY_LABELS.get(family, family),
            "demi-tons": step,
            "partiels": f"{p_low}:{p_high}",
            "poids %": 100 * weight,
        })
    lines.append("### Poids utilisés\n")
    lines.append(markdown_table(weight_rows, ["famille", "demi-tons", "partiels", "poids %"]))
    lines.append("")

    lines.append("## 3. Deviation curves\n")
    lines.append(
        "Chaque graphe `03_deviation_...` correspond à une famille d'intervalles. "
        "À l'intérieur, chaque courbe correspond à un couple de partiels : par exemple, "
        "dans les octaves, `2:1`, `4:2`, `6:3`, etc."
    )
    lines.append(
        "La colonne `deviation_cents` mesure `1200·log2(partiel_haut / partiel_bas)`. "
        "La colonne `beat_hz` mesure `partiel_haut - partiel_bas` en Hz. "
        "Zéro signifie que le couple de partiels coïncide exactement."
    )

    summary_rows = []
    grouped = interval_deviations.groupby(["interval_label", "partial_ratio", "weight_percent"], sort=False)
    for (label, ratio, weight_percent), grp in grouped:
        summary_rows.append({
            "famille": label,
            "partiels": ratio,
            "poids %": float(weight_percent),
            "RMS cents": rms(grp["deviation_cents"]),
            "max |cents|": float(np.nanmax(np.abs(grp["deviation_cents"].to_numpy(float)))),
            "RMS Hz": rms(grp["beat_hz"]),
        })

    lines.append("### Résumé des déviations après optimisation\n")
    lines.append(markdown_table(summary_rows, [
        "famille", "partiels", "poids %", "RMS cents", "max |cents|", "RMS Hz"
    ]))
    lines.append("")

    lines.append("## Comment lire les résultats\n")
    lines.append(
        "- Si une courbe de déviation reste près de zéro, ce couple de partiels est presque pur.\n"
        "- Si une courbe s'éloigne de zéro, c'est le compromis accepté pour préserver d'autres intervalles plus pondérés.\n"
        "- Si l'octave et la douzième sont toutes deux proches de zéro dans une zone, le compromis balanced-like fonctionne bien dans cette zone.\n"
        "- Si une famille très pondérée est mauvaise partout, les B mesurés sont peut-être bruités ou le lissage de B est trop contraignant."
    )

    report = "\n".join(lines)
    path = outdir / "analysis_report.md"
    path.write_text(report, encoding="utf-8")
    return path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_note_list(args):
    if args.notes:
        midis = [parse_note_name(t) for t in args.notes.split(",")]
    else:
        start = parse_note_name(args.start)
        end = parse_note_name(args.end)
        midis = list(range(start, end + 1))

    return [m for m in midis if PIANO_START <= m <= PIANO_END]


def main():
    parser = argparse.ArgumentParser(
        description="Analyse note par note : inharmonicité, tuning curve balanced-like, deviation curves."
    )

    parser.add_argument(
        "--mode",
        choices=["live", "note", "graphs-only", "from-csv"],
        default="live",
    )
    parser.add_argument(
        "--note",
        default="A4",
        help="Note individuelle à enregistrer en mode note, ex : A4, C#5, F3",
    )
    parser.add_argument("--measurements", default="analyse_piano/measurements.csv")
    parser.add_argument("--outdir", default="analyse_piano")
    parser.add_argument("--seconds", type=float, default=2.8)
    parser.add_argument("--fs", type=int, default=48000)
    parser.add_argument("--start", default="A0")
    parser.add_argument("--end", default="C8")
    parser.add_argument(
        "--notes",
        default="",
        help="Ex : A0,C1,F1,A1,C2,F2,A2,C3,F3,A3,C4,F4,A4,C5,F5,A5,C6,F6,A6,C7,C8",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--deviation-unit",
        choices=["cents", "hz"],
        default="cents",
        help="Unité des graphes de deviation curve. Les CSV contiennent toujours cents et Hz.",
    )
    parser.add_argument(
        "--smooth-weight",
        type=float,
        default=0.08,
        help="Poids du lissage de la tuning curve. Plus haut = courbe plus lisse.",
    )
    parser.add_argument(
        "--anchor-weight",
        type=float,
        default=12.0,
        help="Poids de l'ancrage A4 = 0 cent.",
    )

    args = parser.parse_args()
    outdir = Path(args.outdir)

    if args.mode == "note":
        measure_single_note(
            args.note,
            args.seconds,
            args.fs,
            outdir,
            device=args.device,
        )
        return

    if args.mode == "live":
        midis = parse_note_list(args)
        measurements, partials = measure_live(
            midis,
            args.seconds,
            args.fs,
            outdir,
            device=args.device,
        )
    elif args.mode in ["graphs-only", "from-csv"]:
        measurements = pd.read_csv(args.measurements)
    else:
        raise ValueError(f"Mode inconnu : {args.mode}")

    tuning, interval_deviations, B_fit_info = optimize_tuning_curve(
        measurements,
        intervals=BALANCED_LIKE_INTERVALS,
        smooth_weight=args.smooth_weight,
        anchor_weight=args.anchor_weight,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(outdir / "tuning_curve.csv", index=False)
    interval_deviations.to_csv(outdir / "interval_deviations.csv", index=False)

    deviation_files = plot_results(
        tuning,
        interval_deviations,
        outdir,
        intervals=BALANCED_LIKE_INTERVALS,
        deviation_unit=args.deviation_unit,
    )

    report_path = write_analysis_report(
        tuning,
        interval_deviations,
        outdir,
        intervals=BALANCED_LIKE_INTERVALS,
        B_fit_info=B_fit_info,
    )

    print("\nFichiers générés :")

    fixed_files = [
        outdir / "measurements.csv",
        outdir / "partials.csv",
        outdir / "tuning_curve.csv",
        outdir / "interval_deviations.csv",
        outdir / "01_inharmonicite_B.png",
        outdir / "02_tuning_curve_balanced_like.png",
        report_path,
    ]

    for path in fixed_files + deviation_files:
        if path.exists():
            print(f"  - {path}")


if __name__ == "__main__":
    main()