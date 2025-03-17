exp_dir="../experiments/exp01"
for entry in "$exp_dir"/run*; do
    echo "$entry";
    sbatch "$entry"/submit_slurm.sh;
done