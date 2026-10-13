"""smoothed_check.py -- rerun the double-integration net-displacement check
with both the GRF-derived acceleration AND the actual COM reference
low-pass filtered at config.id_lowpass_hz (4.5 Hz, the same cutoff already
used elsewhere in this project for COM/momentum signals -- see
opensim_env.center_of_mass_and_momentum_rate) before integrating/
differentiating, to test whether the large displacement errors found on
real trials are a noise-amplification artifact (would improve with
smoothing) or a genuine low-frequency shape/magnitude mismatch in the GRF
signal during push-off/landing (would NOT improve with smoothing).

Run (from code/, in the opensimPy environment):
    python expt_temp/grf_double_integration_check/smoothed_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from config import DEFAULT  # noqa: E402
from discovery import TrialDiscovery  # noqa: E402
from activities import activity_for  # noqa: E402
from subject import Subject  # noqa: E402
from opensim_env import lowpass  # noqa: E402

SUBJECT_IDS = ("AB03", "AB05", "AB07", "AB08", "AB10", "NTS01")
ACTIVITY_NAME = "single_leg_jump"
OUT_ROOT = Path(__file__).resolve().parent / "results"


def net_displacement_error(time, grf_N, com_m, mass_kg, gravity_mps2, smooth: bool):
    gravity_vec = np.array([0.0, -gravity_mps2, 0.0])
    if smooth:
        grf_N = lowpass(grf_N, time, DEFAULT.id_lowpass_hz)
        com_m = lowpass(com_m, time, DEFAULT.id_lowpass_hz)
    acceleration = grf_N / mass_kg + gravity_vec
    dt0 = time[1] - time[0]
    v0 = (com_m[1] - com_m[0]) / dt0
    velocity = v0 + cumulative_trapezoid(acceleration, time, axis=0, initial=0)
    position = com_m[0] + cumulative_trapezoid(velocity, time, axis=0, initial=0)
    actual_disp = com_m[-1] - com_m[0]
    recon_disp = position[-1] - com_m[0]
    return recon_disp - actual_disp


def main():
    discovery = TrialDiscovery(DEFAULT)
    activity = activity_for(ACTIVITY_NAME, DEFAULT)

    rows = []
    for subject_id in SUBJECT_IDS:
        subject = Subject(subject_id, DEFAULT)
        trials = discovery.trials(subject, activity, "all", for_analysis=True)
        for trial in trials:
            package = trial.output / "trial_data.npz"
            if not package.is_file():
                continue
            data = np.load(package, allow_pickle=True)
            if bool(data.get("quality_omit", False)):
                continue
            time = data["time"].astype(float)
            grf_N = data["grf_N"].astype(float)
            com_m = data["com_m"].astype(float)
            mass_kg = float(data["mass_kg"])

            raw_error = net_displacement_error(time, grf_N, com_m, mass_kg, DEFAULT.gravity_mps2, smooth=False)
            smooth_error = net_displacement_error(time, grf_N, com_m, mass_kg, DEFAULT.gravity_mps2, smooth=True)
            rows.append({
                "Subject": subject_id, "Trial": trial.name,
                "Fy_RawError_mm": round(1000 * raw_error[1], 1),
                "Fy_SmoothedError_mm": round(1000 * smooth_error[1], 1),
            })

    table = pd.DataFrame(rows)
    out_path = OUT_ROOT / "smoothed_vs_raw_comparison.csv"
    table.to_csv(out_path, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved -> {out_path}\n")

    summary = table.groupby("Subject").agg(
        RawMeanAbs_mm=("Fy_RawError_mm", lambda s: s.abs().mean()),
        SmoothedMeanAbs_mm=("Fy_SmoothedError_mm", lambda s: s.abs().mean()),
    )
    summary["Improved"] = summary["SmoothedMeanAbs_mm"] < summary["RawMeanAbs_mm"]
    print(summary.to_string())
    print(f"\nOverall: raw mean|err|={table.Fy_RawError_mm.abs().mean():.1f}mm, "
          f"smoothed mean|err|={table.Fy_SmoothedError_mm.abs().mean():.1f}mm")


if __name__ == "__main__":
    main()
