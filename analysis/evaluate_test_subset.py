from openff.units import unit


# Define the reference energies for each element in kJ/mol
ase = {
    "H": -1574.9057945240868 * unit.kilojoule_per_mole,
    "C": -100086.51170377462 * unit.kilojoule_per_mole,
    "N": -143801.51483853368 * unit.kilojoule_per_mole,
    "O": -197604.4180833407 * unit.kilojoule_per_mole,
    "F": -262248.6420523814 * unit.kilojoule_per_mole,
    "P": -896364.0735122316 * unit.kilojoule_per_mole,
    "S": -1045654.468887562 * unit.kilojoule_per_mole,
    "Cl": -1208514.2106607035 * unit.kilojoule_per_mole,
    "Fe": -3318371.4922360503 * unit.kilojoule_per_mole,
    "Cu": -4307496.132092452 * unit.kilojoule_per_mole,
    "Zn": -4672271.154058549 * unit.kilojoule_per_mole,
    "Br": -6759287.787427634 * unit.kilojoule_per_mole,
    "Pd": -335891.6546589908 * unit.kilojoule_per_mole,
}

# define the dataset statistics to use if the dataset was normalized

dataset_statistic = {
    "training_dataset_statistics": {
        "per_atom_energy_mean": -5.146267064064403,
        "per_atom_energy_stddev": 6.048297128174876,
    }
}


# This function calculates the reference energy for a molecule based on its atomic numbers and the reference energies defined above.
def calculate_reference_energy(atomic_numbers, ase):
    from modelforge.dataset.utils import _ATOMIC_NUMBER_TO_ELEMENT

    atomic_numbers = list(atomic_numbers.reshape(-1))
    # sum up the reference energy for each element in the atomic numbers
    reference_energy = [
        ase[_ATOMIC_NUMBER_TO_ELEMENT[atomic_number]]
        for atomic_number in atomic_numbers
    ]

    return sum(reference_energy)


from modelforge.potential.potential import load_inference_model_from_checkpoint

import torch


filename = "/experiments/exp08_3/run_sm135/run077/logs/aimnet2_tmqm_openff/bqftyw6v/checkpoints/best_AimNet2-tmqm_openff_local-epoch=67.ckpt"

potential = load_inference_model_from_checkpoint(filename, jit=False)
potential.to(device="cuda" if torch.cuda.is_available() else "cpu")

# if the dataset was normalized we need to unnormalize the energy
# to compare with the reference energies in the hdf5 dataset
unnormalize_energy = False

import h5py

# define the input file name of the hdf5 file that contains the fixed test subset
input_filename = "/Users/syan/workdir/modelforge-experiments/experiments/exp08_3/cache/fixed_test_subset/fixed_test_subset_v1.2.hdf5"


from tqdm import tqdm
from dataclasses import dataclass
from modelforge.dataset.utils import _ATOMIC_NUMBER_TO_ELEMENT
from modelforge.utils.prop import NNPInput


energy_diff = []
energy_ref = []
energy_pred = []


snapshots = []
molecule_names = []
with h5py.File(input_filename, "r") as f:
    keys = list(f.keys())

    for key in tqdm(keys):

        # grab all the data from the hdf5 file
        # we could certainly write some helper functions to do this better
        # e.g., loading this into a SourceDataset class instance and accessing
        # each record by key
        atomic_numbers = f[key]["atomic_numbers"][()]
        positions = f[key]["positions"][()]
        energy_key = "dft_total_energy"
        energy = f[key][energy_key][()] * unit.Unit(
            f[key][energy_key].attrs["u"]
        ) - calculate_reference_energy(atomic_numbers, ase)
        total_charge = f[key]["total_charge"][()]
        number_of_atoms = atomic_numbers.shape[0]
        n_configs = f[key]["n_configs"][()]
        spin_multiplicity = f[key]["per_system_spin_multiplicity"][()]
        from modelforge.utils.prop import NNPInput

        for n_config in range(n_configs):
            # print(f"Processing config {n_config} of {n_configs}")

            # could create a helper function to convert a record to NNPInput based on the properties association dict in
            # the toml files.
            nnp_input = NNPInput(
                atomic_numbers=torch.tensor(
                    atomic_numbers.squeeze(), dtype=torch.int32
                ),
                positions=torch.tensor(
                    positions[n_config].reshape(-1, 3), dtype=torch.float32
                ),
                per_system_total_charge=torch.tensor(
                    total_charge[n_config].reshape(-1, 1), dtype=torch.float32
                ),
                atomic_subsystem_indices=torch.zeros(
                    number_of_atoms, dtype=torch.int32
                ),
                per_system_spin_state=torch.tensor(
                    spin_multiplicity[n_config].reshape(-1, 1), dtype=torch.float32
                ),
            )
            nnp_input.to_device(device="cuda" if torch.cuda.is_available() else "cpu")
            molecule_names.append(key)
            output = potential(nnp_input)

            energy_temp = (
                output["per_system_energy"].cpu().detach().numpy().reshape(-1)[0]
            )
            if unnormalize_energy:
                energy_temp = (
                    energy_temp
                    * dataset_statistic["training_dataset_statistics"][
                        "per_atom_energy_stddev"
                    ]
                    + dataset_statistic["training_dataset_statistics"][
                        "per_atom_energy_mean"
                    ]
                )
            energy_diff.append(float((energy_temp - energy[n_config].m).reshape(-1)[0]))
            energy_pred.append(float(energy_temp.reshape(-1)[0]))
            energy_ref.append(float(energy[n_config].m.reshape(-1)[0]))


from matplotlib import pyplot as plt
import numpy as np

# plot the energy
plt.plot(energy_ref, energy_pred, "o")

# add a line y=x
plt.plot(
    [min(min(energy_ref), min(energy_pred)), max(max(energy_ref), max(energy_pred))],
    [min(min(energy_ref), min(energy_pred)), max(max(energy_ref), max(energy_pred))],
)

plt.xlabel("Reference Energy")
plt.ylabel("Predicted Energy")
#
plt.savefig(f"energy_ref_vs_pred.png")
plt.show()
plt.cla()

# create histogram on a log scale of energy difference
plt.hist(energy_diff, bins=100)
plt.yscale("log")
# plt.show()
plt.savefig(f"energy_diff_hist_.png")
plt.show()
plt.cla()

# save to file the data include the name of the molecule
with open(f"energy_diff.txt", "w") as f:
    for i in range(len(energy_diff)):
        f.write(
            f"{molecule_names[i]}\t{energy_ref[i]}\t{energy_pred[i]}\t{energy_diff[i]}\n"
        )

# calculate MAE for the system
mae = sum(abs(np.array(energy_diff))) / len(energy_diff)
print(f"MAE: {mae:.4f} kJ/mol")
