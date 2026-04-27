# Piano Inharmonicity & Balanced Tuning Curve

Measure piano notes one by one, estimate their inharmonicity, then generate a balanced-like piano tuning curve.

## What this tool does

This script can:

- record isolated piano notes from a microphone;
- detect partials for each note;
- estimate the fundamental frequency `f0` and inharmonicity coefficient `B`;
- fit a smooth inharmonicity curve across the keyboard;
- compute a full 88-note tuning curve from A0 to C8;
- optimize a **balanced-like** compromise between octaves, twelfths, fifths, fourths, double octaves, and nineteenth intervals;
- generate CSV files, scientific plots, and a Markdown analysis report.

## Example with mine

<img width="1965" height="1100" alt="01_inharmonicite_B" src="https://github.com/user-attachments/assets/b2730c2a-871a-45f4-a795-2fc580b2b3e6" />

<img width="2091" height="1251" alt="02_tuning_curve_balanced_like" src="https://github.com/user-attachments/assets/e50f56bb-586d-4956-8aab-ae40927328a4" />

<img width="2060" height="1220" alt="03_deviation_octave_cents" src="https://github.com/user-attachments/assets/869ccf6a-6b70-4453-9954-49e0865a1425" />

<img width="2060" height="1220" alt="03_deviation_douzieme_cents" src="https://github.com/user-attachments/assets/2543ca72-f949-452e-b56f-92e04b0bd50a" />

## Why piano tuning needs this

A real piano string is not perfectly harmonic. Because of string stiffness, the partials are slightly sharper than exact integer multiples of the fundamental. This is called *inharmonicity*.

That means a mathematically equal-tempered tuning is not enough: if every first partial is placed exactly on equal temperament, upper partials will not align cleanly, especially in octaves and larger intervals. Piano tuners therefore stretch the tuning, usually lowering the bass and raising the treble compared with equal temperament.

## Quick start

### 1. Record and analyze notes live

```bash
python main.py --mode live --outdir analyse_piano
```

By default, the script records from `A0` to `C8`.

During recording, the program asks you to prepare each note, press Enter, then play immediately.

### 2. Analyze only selected notes

```bash
python main.py --mode live \
  --notes A0,C1,F1,A1,C2,F2,A2,C3,F3,A3,C4,F4,A4,C5,F5,A5,C6,F6,A6,C7,C8 \
  --outdir analyse_piano
```

This is useful for building a first model quickly without recording all 88 notes.

### 3. Replace one note

```bash
python main.py --mode note --note A4 --outdir analyse_piano
```

This records one note, replaces its previous measurement in `measurements.csv`, updates `partials.csv`, and stops without regenerating plots.

### 4. Regenerate graphs from an existing CSV

```bash
python main.py --mode graphs-only \
  --measurements analyse_piano/measurements.csv \
  --outdir analyse_piano
```

`from-csv` behaves similarly:

```bash
python main.py --mode from-csv \
  --measurements analyse_piano/measurements.csv \
  --outdir analyse_piano
```

A good first pass is to record every 3 to 6 semitones, then fill in missing or unstable zones.

## Command-line options

| Option | Default | Description |
|---|---:|---|
| `--mode` | `live` | `live`, `note`, `graphs-only`, or `from-csv` |
| `--note` | `A4` | Single note used in `note` mode |
| `--measurements` | `analyse_piano/measurements.csv` | Input CSV for graph generation |
| `--outdir` | `analyse_piano` | Output directory |
| `--seconds` | `2.8` | Recording duration per note |
| `--fs` | `48000` | Sampling rate |
| `--start` | `A0` | First note when using a range |
| `--end` | `C8` | Last note when using a range |
| `--notes` | empty | Comma-separated note list, for example `A0,C1,F1,A1` |
| `--device` | `None` | Optional audio input device |
| `--deviation-unit` | `cents` | Plot deviation curves in `cents` or `hz` |
| `--smooth-weight` | `0.08` | Higher value produces a smoother tuning curve |
| `--anchor-weight` | `12.0` | Strength of the A4 = 0 cent anchor |

---

## Generated files

The default output directory is `analyse_piano/`.

| File | Description |
|---|---|
| `measurements.csv` | One row per measured note: `f0`, `B`, cents vs equal temperament, fit quality, number of partials |
| `partials.csv` | Detected partials for every measured note |
| `tuning_curve.csv` | Final target frequency and target cents for every piano key |
| `interval_deviations.csv` | Interval deviation data for all interval families and partial ratios |
| `01_inharmonicite_B.png` | Measured and smoothed inharmonicity coefficient `B` |
| `02_tuning_curve_balanced_like.png` | Optimized balanced-like tuning curve |
| `03_deviation_*.png` | Deviation curves by interval family |
| `analysis_report.md` | Auto-generated explanation and summary tables |

The CSV files always contain both cents and Hz deviation data. The option `--deviation-unit` only changes the generated deviation plots.

## Mathematical model

### Equal temperament reference

For MIDI note `m`, equal temperament is computed from A4 = 440 Hz:

```math
F_{\mathrm{ET}}(m) = 440 \cdot 2^{(m - 69)/12}
```

### Inharmonic partial frequency

For partial rank `k`, fundamental-like frequency `F`, and inharmonicity coefficient `B`:

```math
f_k = k F \sqrt{1 + B k^2}
```

This is the core model used to estimate inharmonicity from detected partials.

### Measured deviation from equal temperament

```math
d(m) = 1200 \log_2 \left(\frac{f_0(m)}{F_{\mathrm{ET}}(m)}\right)
```

The result is expressed in cents.

### Smooth inharmonicity curve

Measured `B` values are smoothed using:

```math
\log_{10}(B) = a + b z + c z^2
```

with:

```math
z = \frac{m - 69}{39}
```

and the constraint:

```math
c \ge 0
```

This produces a globally convex inharmonicity curve.

### Interval partial deviation

For an interval between a lower note and an upper note, the deviation is measured by comparing selected partials:

```math
\Delta_{\mathrm{cents}} =
1200 \log_2
\left(
\frac{f_{\mathrm{upper\ partial}}}
     {f_{\mathrm{lower\ partial}}}
\right)
```

and the beat difference is:

```math
\Delta_{\mathrm{Hz}} =
f_{\mathrm{upper\ partial}} - f_{\mathrm{lower\ partial}}
```

A value close to zero means the selected partials nearly coincide.

### Tuning optimization objective

The tuning curve is computed by optimizing note offsets in cents.

The objective contains:

```text
weighted interval partial deviations
+ tuning-curve smoothness
+ A4 anchoring
```

In simplified form:

```math
\min_x
\sum_i w_i \Delta_i(x)^2
+
\lambda \sum_m \left(\Delta^2 x_m\right)^2
+
\alpha x_{A4}^2
```

where:

- `x_m` is the tuning offset of MIDI note `m`;
- `w_i` is the interval weight;
- `λ` is controlled by `--smooth-weight`;
- `α` is controlled by `--anchor-weight`;
- `x_A4` is constrained toward 0 cents.

---

## Balanced-like style

The default style is called **balanced-like** because it gives strong importance to both octaves and twelfths, while keeping secondary intervals in the objective.

Important: the weights are not measured from audio. They define the target style that the optimizer tries to satisfy.

| Family | Semitones | Partial ratio | Weight |
|---|---:|---:|---:|
| Octave | 12 | 2:1 | 80% |
| Octave | 12 | 4:2 | 100% |
| Octave | 12 | 6:3 | 75% |
| Octave | 12 | 8:4 | 45% |
| Octave | 12 | 10:5 | 25% |
| Twelfth | 19 | 3:1 | 100% |
| Twelfth | 19 | 6:2 | 65% |
| Twelfth | 19 | 9:3 | 35% |
| Fifth | 7 | 3:2 | 20% |
| Fifth | 7 | 6:4 | 10% |
| Fourth | 5 | 4:3 | 10% |
| Double octave | 24 | 4:1 | 30% |
| Double octave | 24 | 8:2 | 18% |
| Nineteenth | 31 | 6:1 | 20% |

The optimizer does not choose these weights. It only finds the tuning curve that best satisfies them.

## Scientific reference

More details on the formulas used:

[Rigaud, F., David, B., & Daudet, L. — “A parametric model and estimation techniques for the inharmonicity and tuning of the piano”](https://www.institut-langevin.espci.fr/biblio/2020/3/5/916/files/2013_a_parametric_model_and_estimation_techniques_for_the_inharmonicity_and_tuning_of_the_piano.pdf)

The implementation here is not an official implementation of the paper. It uses a practical note-by-note workflow and a customizable interval-weighted optimization style.

## License
MIT
