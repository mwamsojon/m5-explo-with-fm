"""
Gap 3 — CL sensitivity curve per dept from v13 Optuna DB
=========================================================
Queries the v13 HPO study (200 trials) and extracts, for each dept,
the distribution of geo-mean objective values grouped by that dept's CL.

Output: per-dept table of {CL: [best, median, mean, n_trials]} and a
marginal CL effect showing how strongly each dept's outcome depends on CL.

No new inference needed — uses existing trial data only.

Run:
    cd /home/nmwamsojo/tsfm-explo
    source .venv/bin/activate
    python experiments/13_chronos_cl_sensitivity.py | tee /mnt/lab/nmwamsojo/gap3_cl_sensitivity.log
"""

import json
import os
import numpy as np
import optuna
import warnings
warnings.filterwarnings("ignore")

LAB_DIR     = os.getenv("M5_LAB_DIR", "/mnt/lab/nmwamsojo")
DB_PATH     = LAB_DIR + "/optuna_hpo_dept_v13.db"
DEPT_IDS  = ["FOODS_1","FOODS_2","FOODS_3","HOBBIES_1","HOBBIES_2","HOUSEHOLD_1","HOUSEHOLD_2"]
CL_VALUES = [1, 7, 14, 21, 28]
RESULT_PATH = LAB_DIR + "/gap3_cl_sensitivity_result.json"

def _log(msg): print(msg, flush=True)


if __name__ == "__main__":
    study = optuna.load_study(study_name=None, storage=f"sqlite:///{DB_PATH}")
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    _log(f"Loaded {len(trials)} completed trials from {DB_PATH}")
    _log(f"Best trial: #{study.best_trial.number}  geo={study.best_trial.value:.4f}")
    _log("")

    results = {}

    _log(f"{'Dept':<14}  {'CL':>3}  {'n':>4}  {'best':>7}  {'median':>7}  {'mean':>7}  {'worst':>7}")
    _log("-" * 65)

    for dept in DEPT_IDS:
        dept_results = {}
        for cl in CL_VALUES:
            matching = [t.value for t in trials if t.params.get(dept) == cl]
            if not matching:
                continue
            dept_results[cl] = {
                "n":      len(matching),
                "best":   float(np.min(matching)),
                "median": float(np.median(matching)),
                "mean":   float(np.mean(matching)),
                "worst":  float(np.max(matching)),
                "values": sorted(matching),
            }
            _log(f"{dept:<14}  {cl:>3}  {len(matching):>4}  "
                 f"{np.min(matching):>7.4f}  {np.median(matching):>7.4f}  "
                 f"{np.mean(matching):>7.4f}  {np.max(matching):>7.4f}")

        results[dept] = dept_results
        _log("")

    # Per-dept sensitivity: range of best-per-CL values
    _log("="*65)
    _log("SENSITIVITY SUMMARY — range of best values across CL options")
    _log(f"{'Dept':<14}  {'best_CL':>7}  {'best_val':>8}  {'worst_CL':>8}  {'worst_val':>9}  {'range':>7}  {'verdict'}")
    _log("-" * 80)

    sensitivity = {}
    for dept in DEPT_IDS:
        dr = results[dept]
        if not dr:
            continue
        best_by_cl   = {cl: v["best"] for cl, v in dr.items()}
        best_cl      = min(best_by_cl, key=best_by_cl.get)
        worst_cl     = max(best_by_cl, key=best_by_cl.get)
        best_val     = best_by_cl[best_cl]
        worst_val    = best_by_cl[worst_cl]
        cl_range     = worst_val - best_val

        # Verdict: how much does CL choice matter?
        if cl_range < 0.002:
            verdict = "insensitive (CL doesn't matter)"
        elif cl_range < 0.005:
            verdict = "mildly sensitive"
        elif cl_range < 0.015:
            verdict = "moderately sensitive"
        else:
            verdict = "HIGHLY sensitive"

        _log(f"{dept:<14}  {best_cl:>7}  {best_val:>8.4f}  {worst_cl:>8}  "
             f"{worst_val:>9.4f}  {cl_range:>7.4f}  {verdict}")

        sensitivity[dept] = {
            "best_cl": best_cl, "best_val": best_val,
            "worst_cl": worst_cl, "worst_val": worst_val,
            "range": cl_range, "verdict": verdict,
        }

    results["sensitivity"] = sensitivity

    # Cross-dept view: for each CL, which dept benefits most from it vs others?
    _log("\n" + "="*65)
    _log("CROSS-DEPT VIEW — best marginal CL per dept")
    _log("(shows which CLs each dept 'wants', marginalised over other depts' CLs)")
    for dept in DEPT_IDS:
        if dept not in results or not results[dept]:
            continue
        sens = sensitivity.get(dept, {})
        _log(f"  {dept:<14}: optimal CL={sens.get('best_cl','?')}  "
             f"range={sens.get('range',0):.4f}  {sens.get('verdict','')}")

    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    _log(f"\nResults saved → {RESULT_PATH}")
