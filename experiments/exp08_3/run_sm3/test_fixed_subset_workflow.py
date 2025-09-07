import glob
import os
import sys
sys.path.append('../../../analysis')

import toml

import helper


dataset_filename = f"../cache/fixed_test_subset/fixed_test_subset_sm_3_v1.2.hdf5"
ckpt_list = sorted(glob.glob(f"./run*/logs/aimnet2_tmqm_openff/*/checkpoints/*"))
# print(ckpt_list)

# do tests with each checkpoint file
for i, ckpt in enumerate(ckpt_list):
    print(i, ckpt)

    # only test the last checkpoint available
    run_log_path = os.path.dirname(
        os.path.dirname(ckpt)
    )
    if i == len(ckpt_list) - 1:
        pass
    else:
        if run_log_path == os.path.dirname(
            os.path.dirname(ckpt_list[i + 1])
        ):
            continue

    run_id_path = os.path.dirname(
        os.path.dirname(
            os.path.dirname(run_log_path)
        )
    )

    save_dir = os.path.join(
        run_id_path,
        "test_results",
        os.path.splitext(os.path.basename(ckpt))[0]
    )

    # find experiment name
    config = toml.load(
        os.path.join(run_id_path, "config.toml")
    )
    experiment_name = config["runtime"]["experiment_name"]

    helper.test_nnp_with_fixed_tmqm_subset(
        ckpt,
        dataset_filename,
        save_dir,
        experiment_name,
    )
