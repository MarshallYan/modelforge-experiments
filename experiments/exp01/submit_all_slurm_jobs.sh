exp_dir="."
for entry in "$exp_dir"/run*; do
    echo "$entry";
    sbatch "$entry"/submit_slurm.sh;
done