#!/bin/bash
# Runs weight configs 1-3 sequentially for the pilot trial
# (AB03_NOEXO_SLJ_R_1, side R, push-off window [3.24, 3.84]). Config 0
# (baseline) is NOT rerun -- reused directly from
# ../opensimad_gain_check/results/AB03_NOEXO_SLJ_R_1_pushoff/GRF.csv.
set -e

CONDA="/c/Users/sjogleka/AppData/Local/anaconda3/condabin/conda.bat"
RUNNER="/g/Shared drives/SuhrudLib2/Projects/Project_1_RTS_ASSESSMENT/code/expt_temp/OpensimAD_weights_optimization/run_weighted_trial.py"
RESULTS="/g/Shared drives/SuhrudLib2/Projects/Project_1_RTS_ASSESSMENT/code/expt_temp/OpensimAD_weights_optimization/results"
SESSION="OpenCapData_5d72f488-5fe2-4e98-b0ee-0ac018f834d4"
TRIAL="ABO3_NOEXO_SLJ_R_1"
WSTART="3.24"
WEND="3.84"

run_config() {
  config_name=$1; weights_json=$2; case=$3
  mkdir -p "$RESULTS/${config_name}"
  echo "=== $config_name: $weights_json ==="
  "$CONDA" run -n opencap-processing-grf --no-capture-output python "$RUNNER" \
    "$SESSION" "$TRIAL" "$WSTART" "$WEND" "$case" "$weights_json" \
    > "$RESULTS/${config_name}/run.log" 2>&1
  echo "exit_code=$? for $config_name"
  src="/c/Users/sjogleka/Documents/opencap-processing-grf-jumptest/Data/$SESSION/OpenSimData/Dynamics/$TRIAL/Results_${case}/GRF.csv"
  if [ -f "$src" ]; then
    cp "$src" "$RESULTS/${config_name}/GRF.csv"
    echo "copied GRF.csv for $config_name"
  else
    src_nc="/c/Users/sjogleka/Documents/opencap-processing-grf-jumptest/Data/$SESSION/OpenSimData/Dynamics/$TRIAL/Results_${case}_NonConvergent/GRF.csv"
    if [ -f "$src_nc" ]; then
      cp "$src_nc" "$RESULTS/${config_name}/GRF_NonConvergent.csv"
      echo "copied GRF_NonConvergent.csv for $config_name"
    else
      echo "WARNING: no GRF.csv found for $config_name"
    fi
  fi
}

run_config "config1_accel500" '{"accelerationTrackingTerm": 500}' "ab03_r1_w_config1"
run_config "config2_activation01" '{"activationTerm": 0.1}' "ab03_r1_w_config2"
run_config "config3_combined" '{"accelerationTrackingTerm": 500, "activationTerm": 0.1}' "ab03_r1_w_config3"

echo "ALL_CONFIGS_DONE"
