#!/bin/bash
# Submit all three layers as a DAG.
# Layer 1 and 2 run in parallel; Layer 3 waits for both to succeed.

set -e
mkdir -p logs data/checkpoints

jid1=$(sbatch --parsable slurm/layer1.sh)
echo "Layer 1 submitted: job $jid1"

jid2=$(sbatch --parsable slurm/layer2.sh)
echo "Layer 2 submitted: job $jid2"

jid3=$(sbatch --parsable \
       --dependency=afterok:${jid1}:${jid2} \
       slurm/layer3.sh)
echo "Layer 3 submitted: job $jid3 (depends on $jid1, $jid2)"

echo ""
echo "Monitor with:  squeue -j ${jid1},${jid2},${jid3}"
