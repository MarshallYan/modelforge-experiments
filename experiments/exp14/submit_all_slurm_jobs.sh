run_dir="./runs"
for entry in "$run_dir"/run*; do
    echo "$entry";
    sbatch "$entry"/submit_slurm.sh;
done