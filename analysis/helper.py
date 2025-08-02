import re
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional

import numpy as np
import pandas as pd
import torch

from modelforge.dataset.dataset import TorchDataset, initialize_datamodule
from modelforge.dataset.utils import RandomRecordSplittingStrategy, FirstComeFirstServeSplittingStrategy, SplittingStrategy
from modelforge.utils.prop import PropertyNames


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
                    except NameError:
                        pass
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






