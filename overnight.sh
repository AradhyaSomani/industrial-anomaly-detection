#!/bin/bash
# Overnight run: auto-detect accuracy, 2 extra feature-AE runs per category, then screw at 512 px.
# Run from the project folder with the venv active:  bash overnight.sh 2>&1 | tee overnight.log

CATS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"

echo "=== 0. Back up the current feature models (the ones the demo uses) ==="
mkdir -p backups
rm -rf backups/checkpoints_feat_main backups/outputs_feat_main
cp -r checkpoints_feat backups/checkpoints_feat_main
cp -r outputs_feat backups/outputs_feat_main

echo "=== 1. Auto-detect accuracy (current models) ==="
python -u src/eval_autodetect.py

echo "=== 2. Two extra training runs per category, for mean +/- std ==="
for run in 2 3; do
  for c in $CATS; do
    echo "--- run $run: $c ---"
    python -u src/train_feat.py --category $c && python -u src/evaluate_feat.py --category $c
    mkdir -p outputs_runs/$c
    cp outputs_feat/$c/results.json outputs_runs/$c/run$run.json
  done
done
# run 1 = the original results, already on disk
for c in $CATS; do cp backups/outputs_feat_main/$c/results.json outputs_runs/$c/run1.json; done

echo "=== 3. Restore the original models so the demo is unchanged ==="
rm -rf checkpoints_feat outputs_feat
cp -r backups/checkpoints_feat_main checkpoints_feat
cp -r backups/outputs_feat_main outputs_feat

echo "=== 4. Screw at 512 px ==="
python -u src/train_feat.py --category screw --size 512 && python -u src/evaluate_feat.py --category screw --size 512

echo "=== Done ==="
