import re
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import toml
import torch
from dataclasses import dataclass
from openff.units import unit
from tqdm import tqdm

from modelforge.dataset.dataset import TorchDataset, initialize_datamodule
from modelforge.dataset.utils import _ATOMIC_NUMBER_TO_ELEMENT
from modelforge.dataset.utils import RandomRecordSplittingStrategy
from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy
from modelforge.dataset.utils import SplittingStrategy
from modelforge.curate.datasets.tmqm_openff_curation import tmQMOpenFFCuration
from modelforge.curate.properties import (
    AtomicNumbers,
    PartialCharges,
    Positions,
    DipoleMomentPerSystem,
    DipoleMomentScalarPerSystem,
)
from modelforge.potential.potential import load_inference_model_from_checkpoint
from modelforge.train.training import CalculateProperties
from modelforge.utils.prop import BatchData, NNPInput, PropertyNames


def extract_config(config_str, key) -> dict:
    key_match = re.search(r'\b'+key+r'\b', config_str)

    if not key_match:
        raise KeyError("No such key in the config string!")
    key_begin_idx = key_match.start()

    key_end_idx = config_str.find(' ', key_begin_idx)
    extracted_string = config_str[key_begin_idx : key_end_idx]

    count = 0
    while count < 1000:
        count += 1

        # match parentheses 
        if extracted_string.count(']') - extracted_string.count('[') < 0 or extracted_string.count(')') - extracted_string.count('(') < 0 or extracted_string.count('}') - extracted_string.count('{') < 0:
            key_end_idx = config_str.find(' ', key_end_idx + 1)
            extracted_string = config_str[key_begin_idx : key_end_idx]

        # get rid of the last ','
        elif extracted_string[-1] == ',':
            extracted_string = extracted_string[:-1]

        # extract dict from the extracted string
        else:
            values = extracted_string[len(key) + 1 : ]
            tail_length = values.count(']') - values.count('[') + values.count(')') - values.count('(') + values.count('}') - values.count('{')
            values = values[:len(values) - tail_length]

            # convert to dict
            try:
                values = int(values)
            except ValueError:
                try:
                    values = float(values)
                except ValueError:
                    try:
                        values = eval(values)
                    except:
                        print(values)
            result = {key: values}
            return result


class DatasetProperties(ABC):
    def __init__(
        self,
        seed: int,
        train_dataset: torch.utils.data.dataset.Subset,
        val_dataset: torch.utils.data.dataset.Subset,
        test_dataset: torch.utils.data.dataset.Subset,
        property_names: list,
        dataset: TorchDataset = None,
    ):
        self.seed = seed
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.property_names = property_names
        self.dataset = dataset if dataset else train_dataset.dataset
        self.dataframe = None

    @abstractmethod
    def collect_property(self):
        pass

    def prepare(self):
        # initialize
        properties = {"dataset_type": [], "seed": []}
        for property_name in self.property_names:
            properties[property_name] = []

        for i, dataset in enumerate([self.train_dataset, self.val_dataset, self.test_dataset]):
            for j, property_name in enumerate(self.property_names):
                collection = self.collect_property(
                    property_name=property_name,
                    dataset=dataset,
                    to_array=True,
                )
                properties[property_name] += collection

                # add same length of train/validation/test tags
                if j == 0:  # only do for the first property
                    if i == 0:
                        properties["dataset_type"] += ["training"] * len(collection)
                    elif i == 1:
                        properties["dataset_type"] += ["validation"] * len(collection)
                    else:
                        properties["dataset_type"] += ["test"] * len(collection)
                    properties["seed"] += [self.seed] * len(collection)
        
        self.dataframe = pd.DataFrame(properties)
        return self.dataframe
        
    def update(self, property_name, to_array=False):
        properties = []
        for i, dataset in enumerate([self.train_dataset, self.val_dataset, self.test_dataset]):
            collection = self.collect_property(
                property_name=property_name,
                dataset=dataset,
                to_array=to_array,
            )
            properties += collection
        self.dataframe[property_name] = properties

        return self.dataframe

    
class PerGeometryProperties(DatasetProperties):
    def collect_property(
        self,
        property_name: str,
        dataset: Union[TorchDataset, torch.utils.data.dataset.Subset] = None,
        to_array: bool = True,
    ) -> list:
        if not dataset:
            dataset = self.dataset
    
        collection = []
        if property_name in dir(dataset[0].nnp_input):
            for geometry in dataset:
                property_value = getattr(geometry.nnp_input, property_name)
                collection.append(property_value)
        elif property_name in dir(dataset[0].metadata):
            for geometry in dataset:
                property_value = getattr(geometry.metadata, property_name)
                collection.append(property_value)
        else:
                raise ValueError(f"property {property_name} does not exist!")
    
        if to_array:
            if collection[0].shape == torch.Size([1]):
                collection = [value.item() for value in collection]
            else:
                collection = [value.numpy() for value in collection]
    
        return collection


class PerCompoundProperties(DatasetProperties):
    def collect_property(
        self,
        property_name: str,
        dataset: Union[TorchDataset, torch.utils.data.dataset.Subset] = None,
        to_array: bool = True,
    ):
        if not dataset:
            dataset = self.dataset
    
        # We assume that the order of geometries aren't shuffled, 
        # meaning that geometries of the same compound are next to each other in order.
        
        if isinstance(dataset, TorchDataset):  # full dataset
            indices_in_full_dataset = range(0, dataset.length)
            series_mol_start_idxs_by_rec = dataset.series_mol_start_idxs_by_rec
        else:  # subset
            indices_in_full_dataset = dataset.indices
            series_mol_start_idxs_by_rec = dataset.dataset.series_mol_start_idxs_by_rec
    
        collection_indices_in_full_dataset = []
        collection = []
    
        if property_name in dir(dataset[0].nnp_input):
            for i, compound in enumerate(dataset):
                index_in_full_dataset = indices_in_full_dataset[i]
                if index_in_full_dataset in series_mol_start_idxs_by_rec or i == 0:
                    property_value = getattr(compound.nnp_input, property_name)
                    collection_indices_in_full_dataset.append(index_in_full_dataset)
                    collection.append(property_value)
                    last_compound_index_in_full_dataset = index_in_full_dataset
                    pass
                else:
                    pass
                    for j, _ in enumerate(series_mol_start_idxs_by_rec):
                        if _ - index_in_full_dataset > 0:
                            unique_compound_start_index = series_mol_start_idxs_by_rec[j - 1]
                            if last_compound_index_in_full_dataset < unique_compound_start_index:
                                property_value = getattr(compound.nnp_input, property_name)
                                collection_indices_in_full_dataset.append(index_in_full_dataset)
                                collection.append(property_value)
                                last_compound_index_in_full_dataset = index_in_full_dataset
                                break
                            break
                            
        elif property_name in dir(dataset[0].metadata):
            for i, compound in enumerate(dataset):
                index_in_full_dataset = indices_in_full_dataset[i]
                if index_in_full_dataset in series_mol_start_idxs_by_rec or i == 0:
                    property_value = getattr(compound.metadata, property_name)
                    collection_indices_in_full_dataset.append(index_in_full_dataset)
                    collection.append(property_value)
                    last_compound_index_in_full_dataset = index_in_full_dataset
                    pass
                else:
                    pass
                    for j, _ in enumerate(series_mol_start_idxs_by_rec):
                        if _ - index_in_full_dataset > 0:
                            unique_compound_start_index = series_mol_start_idxs_by_rec[j - 1]
                            if last_compound_index_in_full_dataset < unique_compound_start_index:
                                property_value = getattr(compound.metadata, property_name)
                                collection_indices_in_full_dataset.append(index_in_full_dataset)
                                collection.append(property_value)
                                last_compound_index_in_full_dataset = index_in_full_dataset
                                break
                            break
    
        if to_array:
            if collection[0].shape == torch.Size([1]):
                collection = [value.item() for value in collection]
            else:
                collection = [value.numpy() for value in collection]
    
        return collection

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

def test_nnp_with_fixed_tmqm_subset(
        checkpoint_filename,
        dataset_filename,
        save_dir,
        experiment_name,
        unnormalize_energy=False,
):
    os.makedirs(save_dir, exist_ok=True)
    print(f"{save_dir} created.")

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

    # load potential
    potential = load_inference_model_from_checkpoint(checkpoint_filename, jit=False)
    potential.to(device="cuda" if torch.cuda.is_available() else "cpu")

    # load dataset and test the nnp
    energy_diff = []
    energy_ref = []
    energy_pred = []
    dipole_moment_diff = []
    dipole_moment_ref = []
    dipole_moment_pred = []
    partial_charge_diff = []
    partial_charge_ref = []
    partial_charge_pred = []

    molecule_names = []
    with h5py.File(dataset_filename, "r") as f:
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
            dipole_moment = f[key]["scf_dipole"][()]
            partial_charge = f[key]["lowdin_partial_charges"][()]

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

                # test energy results
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

                # test partial charge results
                partial_charge_temps = output["per_atom_charge"].cpu().detach().numpy().reshape(-1)
                partial_charge_diff.append((partial_charge_temps - partial_charge[n_config]).reshape(-1))
                partial_charge_pred.append(partial_charge_temps.reshape(-1))
                partial_charge_ref.append(partial_charge[n_config].reshape(-1))

                # test dipole moment results
                dc = tmQMOpenFFCuration("tmqm_openff")
                dipole_moment_temp = dc.compute_dipole_moment(
                    atomic_numbers=AtomicNumbers(value=nnp_input.atomic_numbers.cpu().detach().numpy().reshape(-1, 1)),
                    partial_charges=PartialCharges(
                        value=partial_charge_temps.reshape(1, -1, 1),
                        units=unit.Unit(f[key]["lowdin_partial_charges"].attrs["u"])
                    ),
                    positions=Positions(
                        value=nnp_input.positions.cpu().detach().numpy().reshape(1, -1, 3),
                        units=unit.Unit(f[key]["positions"].attrs["u"])
                    ),
                ).value

                dipole_moment_diff.append(list(dipole_moment_temp - dipole_moment[n_config]))
                dipole_moment_pred.append(list(dipole_moment_temp))
                dipole_moment_ref.append(list(dipole_moment[n_config]))

    # plot and save
    plot_predictions_vs_reference(
        energy_ref,
        energy_pred,
        save_dir,
        "energy_pred_vs_ref_plot.png",
        "Reference Energy (kJ/mol)",
        "Predicted Energy (kJ/mol)",
    )
    plot_histogram(
        energy_diff,
        save_dir,
        "energy_histogram.png",
        "Energy Difference (kJ/mol)",
        "Counts",
    )

    # save to file the data include the name of the molecule
    with open(os.path.join(save_dir, "ref_pred_diff.txt"), "w") as f:
        f.write("molecule_names\tenergy_ref\tenergy_pred\tenergy_diff\n")
        for i in range(len(energy_diff)):
            f.write(
                f"{molecule_names[i]}\t{energy_ref[i]}\t{energy_pred[i]}\t{energy_diff[i]}\n"
            )

    # calculate MAE for the system and save
    energy_mae = np.sum(abs(np.array(energy_diff))) / len(energy_diff)
    energy_rmse = np.sqrt(np.sum((np.array(energy_diff))**2) / len(energy_diff))


    partial_charge_diff = np.concatenate(partial_charge_diff)
    partial_charge_mae = np.sum(abs(np.array(partial_charge_diff))) / len(partial_charge_diff)
    partial_charge_rmse = np.sqrt(np.sum((np.array(partial_charge_diff))**2) / len(partial_charge_diff))


    dipole_moment_diff = np.array(dipole_moment_diff).reshape(-1)
    dipole_moment_mae = np.sum(abs(np.array(dipole_moment_diff))) / len(dipole_moment_diff)
    dipole_moment_rmse = np.sqrt(np.sum((np.array(dipole_moment_diff))**2) / len(dipole_moment_diff))


    # partial_charge_mae = 0
    # partial_charge_rmse = 0
    # dipole_moment_mae = 0
    # dipole_moment_rmse = 0
    # for i in range(len(partial_charge_diff)):
    #     partial_charge_mae += np.linalg.norm(abs(np.array(partial_charge_diff[i])))
    #     partial_charge_rmse += np.linalg.norm(np.array(partial_charge_diff[i]) ** 2)
    # partial_charge_mae /= len(partial_charge_diff)
    # partial_charge_rmse = np.sqrt(partial_charge_rmse / len(partial_charge_diff))
    # for i in range(len(dipole_moment_diff)):
    #     dipole_moment_mae += abs(np.linalg.norm(dipole_moment_pred[i]) - np.linalg.norm(dipole_moment_ref[i]))
    #     dipole_moment_rmse += (np.linalg.norm(dipole_moment_pred[i]) - np.linalg.norm(dipole_moment_ref[i])) ** 2
    # dipole_moment_mae /= len(dipole_moment_diff)
    # dipole_moment_rmse = np.sqrt(dipole_moment_rmse / len(dipole_moment_diff))


    print(f"MAE: {energy_mae:.4f} kJ/mol; {partial_charge_mae:.4f} e; {dipole_moment_mae:.4f} e*nm")
    print(f"RMSE: {energy_rmse:.4f} kJ/mol; {partial_charge_rmse:.4f} e; {dipole_moment_rmse:.4f} e*nm")
    print("============================================================")

    with open(os.path.join(save_dir, "mae.txt"), "w") as f:
        f.write("name\ttest/per_system_energy/mae\ttest/per_system_energy/rmse\ttest/per_atom_charge/mae\ttest/per_atom_charge/rmse\ttest/per_system_dipole_moment/mae\ttest/per_system_dipole_moment/rmse\n")
        f.write(f"{experiment_name}\t{energy_mae}\t{energy_rmse}\t{partial_charge_mae}\t{partial_charge_rmse}\t{dipole_moment_mae}\t{dipole_moment_rmse}\n")

def plot_predictions_vs_reference(ref, pred, save_dir, filename, xlabel, ylabel):
    ax = sns.scatterplot(
        x=ref,
        y=pred,
    )
    ax = sns.lineplot(
        ax=ax,
        x=[min(min(ref), min(pred)), max(max(ref), max(pred))],
        y=[min(min(ref), min(pred)), max(max(ref), max(pred))],
        color='red',
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.figure.savefig(os.path.join(save_dir, filename))
    plt.clf()

def plot_histogram(
        diff,
        save_dir,
        filename,
        xlabel,
        ylabel,
        bins=100,
        log_y_scale=True,
):
    ax = sns.histplot(data=diff, bins=bins)
    if log_y_scale:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.figure.savefig(os.path.join(save_dir, filename))
    plt.clf()
