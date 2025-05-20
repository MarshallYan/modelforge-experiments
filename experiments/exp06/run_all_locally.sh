run_dir="./runs"
for entry in "$run_dir"/run*; do
    cd $entry
    echo "$entry";
    bash run_locally.sh;
    cd ../../
done