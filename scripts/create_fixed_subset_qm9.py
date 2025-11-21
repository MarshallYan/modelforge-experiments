from modelforge.curate import create_dataset_from_hdf5, Record, SourceDataset
from modelforge.curate.properties import *

import numpy as np
from openff.units import unit
from loguru import logger


def calculate_self_energy(atomic_numbers: np.ndarray, ase: dict) -> unit.Quantity:
    """
    Calculate the reference energy for a molecule given its atomic numbers and a dictionary of
    atomic reference energies.

    Parameters
    ----------
    atomic_numbers: np.ndarray
        An array of atomic numbers representing the elements in the molecule.
        Note, this will be flattened to 1D inside the function, so the shape does not matter.
    ase: dict
        A dictionary mapping element symbols to their reference energies.

    Returns
    -------
    reference_energy: unit.Quantity
        The total reference energy for the molecule as a unit.Quantity.
    """
    from modelforge.dataset.utils import _ATOMIC_NUMBER_TO_ELEMENT

    atomic_numbers = list(atomic_numbers.reshape(-1))

    # sum up the reference energy for each element in the atomic numbers
    reference_energy = [
        ase[_ATOMIC_NUMBER_TO_ELEMENT[atomic_number]]
        for atomic_number in atomic_numbers
    ]

    return sum(reference_energy)


# we can define a property map to map string names to property classes
# note, in a future revision, we should consider automating this mapping
# by encoding the property class name in the dataset itself

property_map = {
    "atomic_numbers": AtomicNumbers,
    "positions": Positions,
    "partial_charges": PartialCharges,
    "dipole_moment_per_system": DipoleMomentPerSystem,
    "polarizability": Polarizability,
    "dipole_moment_scalar_per_system": DipoleMomentScalarPerSystem,
    "energy_of_homo": Energies,
    "energy_of_lumo": Energies,
    "lumo-homo_gap": Energies,
    "zero_point_vibrational_energy": Energies,
    "internal_energy_at_0K": Energies,
    "internal_energy_at_298.15K": Energies,
    "enthalpy_at_298.15K": Energies,
    "free_energy_at_298.15K": Energies,
    "internal_energy_at_0K_regressed": Energies,
    "internal_energy_at_0K_regressed_shifted_min": Energies,
    "internal_energy_at_0K_regressed_shifted_max": Energies,
}

# regressed values, copied from the qm9 dataset yaml file
ase_regression = {
    "H": -1313.4668615546 * unit.kilojoule_per_mole,
    "C": -99366.70745535441 * unit.kilojoule_per_mole,
    "N": -143309.9379722722 * unit.kilojoule_per_mole,
    "O": -197082.0671774158 * unit.kilojoule_per_mole,
    "F": -261811.54555874597 * unit.kilojoule_per_mole,
}

# define the input file
# This file comes from: https://zenodo.org/records/17536462
qm9_dataset = create_dataset_from_hdf5(
    hdf5_filename="/Users/syan/workdir/modelforge-experiments/datasets/qm9_dataset_v1.2.hdf5",
    dataset_name="qm9_dataset",
    property_map=property_map,
)

from tqdm import tqdm

record_names = qm9_dataset.keys()

# we will loop over the records, remove the self energy, and store the adjusted energies
# so that we can get the min and max energies and store those shifted values
logger.info("First pass through the dataset to compute adjusted energies.")
energies = []
for record_name in tqdm(record_names):
    # get the record, energy, and atomic numbers
    record = qm9_dataset.get_record(record_name)
    energy = record.get_property("internal_energy_at_0K")
    atomic_numbers = record.get_property("atomic_numbers").value

    self_energy = calculate_self_energy(atomic_numbers, ase_regression)

    adjusted_energy = energy.value * energy.units - self_energy

    # let us make sure we are in kJ/mol
    adjusted_energy = adjusted_energy.to(unit.kilojoule_per_mole, "chem")
    energies.append(adjusted_energy.m)

    regressed_energy_prop = Energies(
        name="internal_energy_at_0K_regressed",
        value=adjusted_energy.m,
        units=adjusted_energy.u,
    )
    record.add_property(regressed_energy_prop)
    qm9_dataset.update_record(record)

max = np.max(np.concatenate(energies))
min = np.min(np.concatenate(energies))


# we will loop over this again to shift the energies by the max/min and add those to the dataset
logger.info("Second pass through the dataset to add shifted energies.")
for record_name in tqdm(record_names):
    record = qm9_dataset.get_record(record_name)
    energy = record.get_property("internal_energy_at_0K_regressed")

    # get the adjust energy and ensure it is in kJ/mol, since we will just deal with the magnitude
    adjusted_energy = (
        (energy.value * energy.units).to(unit.kilojoule_per_mole, "chem").m
    )
    shifted_energy_min = adjusted_energy - min
    shifted_energy_max = max - adjusted_energy

    shifted_energy_min_prop = Energies(
        name="internal_energy_at_0K_regressed_shifted_min",
        value=shifted_energy_min,
        units=unit.kilojoule_per_mole,
    )
    shifted_energy_max_prop = Energies(
        name="internal_energy_at_0K_regressed_shifted_max",
        value=shifted_energy_max,
        units=unit.kilojoule_per_mole,
    )
    record.add_property(shifted_energy_min_prop)
    record.add_property(shifted_energy_max_prop)
    qm9_dataset.update_record(record)

# we will write the dataset to a new hdf5 file
# so we don't have to worry about modifications to the original dataset
# as we generate several fixed subsets

qm9_dataset.to_hdf5(
    file_path="./", file_name="qm9_dataset_with_regressed_energies.hdf5"
)


# now let us create some datasets with fixed test subsets so we can have consistent benchmarking
# while looking at data efficiency
# Let us do a 10% test set, we will use a fixed random seed to ensure reproducibility
# let us loop over 5 different random seeds to create 5 different splits

seeds = [42, 123, 456, 789, 101112]

n_records = qm9_dataset.total_records()

test_fraction = 0.1

for seed in seeds:
    logger.info(f"Creating fixed subset with seed {seed}")
    # read in the dataset again to ensure we start fresh
    qm9_dataset_temp = create_dataset_from_hdf5(
        hdf5_filename="./qm9_dataset_with_regressed_energies.hdf5",
        dataset_name="qm9_dataset",
        property_map=property_map,
    )

    np.random.seed(seed)
    indices = np.arange(n_records)
    # note np.random.shuffle shuffles in place
    np.random.shuffle(indices)

    # let us grab the first chunk for the test set
    test_chunk = int(n_records * test_fraction)
    # just grab the names for the test chunk
    test_record_names = [record_names[i] for i in indices[:test_chunk]]

    # create a new dataset for the test subset
    test_subset = SourceDataset(name=f"qm9_dataset_fixed_test_subset_seed_{seed}")

    # loop over the records and add them to the test subset
    # we will also remove them from the main dataset to create the training/validation set
    for record_name in test_record_names:
        # get the record
        record = qm9_dataset_temp.get_record(record_name)

        # add a copy to the test subset
        test_subset.add_record(record)

        # remove the record from the main dataset, as it is now in the test subset
        qm9_dataset_temp.remove_record(record_name)

    # let us write out the test_subset
    test_subset.to_hdf5(
        file_path="./",
        file_name=f"qm9_dataset_test_subset_seed_{seed}.hdf5",
    )

    # now let us write out the training/validation set
    qm9_dataset_temp.to_hdf5(
        file_path="./", file_name=f"qm9_dataset_trainval_subset_seed_{seed}.hdf5"
    )
