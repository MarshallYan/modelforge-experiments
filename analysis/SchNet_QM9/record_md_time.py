import os
import sys
import time


import openmm
from openmm import LangevinMiddleIntegrator
from openmm import PythonForce
from openmm.app import (
    PDBFile, 
    Simulation, 
    StateDataReporter, 
    DCDReporter,
)
from openmm.unit import (
    kelvin,
    picosecond,
    femtosecond,
    nanometer,
    kilojoules_per_mole,
)
import pandas as pd

from modelforge.openmm.examples.helper_functions import openmm_topology_from_smiles
from modelforge.openmm.potential import generate_compute
from modelforge.potential.potential import load_inference_model_from_checkpoint


def main():
    # make the csv format data
    cols = ["glucose", "phenylalanine", "caffeine", "camptothecin", "nirmatrelvir"]
    df = pd.DataFrame(columns=cols)
    
    # benchmarking molecules
    benchmarking_mols = {
        "glucose": "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O",
        "phenylalanine": "c1ccc(cc1)C[C@H](C(=O)O)N",
        "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "camptothecin": r"O=C\1N4\C(=C/C2=C/1COC(=O)[C@]2(O)CC)c3nc5c(cc3C4)cccc5",
        "nirmatrelvir": "CC1([C@@H]2[C@H]1[C@H](N(C2)C(=O)[C@H](C(C)(C)C)NC(=O)C(F)(F)F)C(=O)N[C@@H](C[C@@H]3CCNC3=O)C#N)C",
    }
    
    checkpoint_dir = '/home/yans3/workdir/shuaiy/modelforge-experiments/experiments/exp14/downloads/checkpoints/'
    for dirpath, dirname, filenames in os.walk(checkpoint_dir):
        for filename in filenames:
            checkpoint_file_path = os.path.join(dirpath, filename)
            model_id = os.path.basename(dirpath)[6:-3]  # model-ms7006xl_v0
    
            # setup potential
            potential = load_inference_model_from_checkpoint(checkpoint_file_path, jit=False)
    
            # setup molecule (loop through)
            for i, mol in enumerate(benchmarking_mols.keys()):
                runtime = []
                for j in range(3):  # duplicates
                    topology, positions = openmm_topology_from_smiles(smiles=benchmarking_mols[mol], optimize=True)
                    atomic_numbers = [atom.element.atomic_number for atom in topology.atoms()]
        
                    comp = generate_compute(potential=potential, atomic_numbers=atomic_numbers)
                    system_force = PythonForce(comp)
        
                    # define the system
                    system = openmm.System()
                    for atom in topology.atoms():
                        system.addParticle(atom.element.mass)
                    
                    # add the system_force instance defined above that wraps modelforge potential for PythonFroce
                    system.addForce(system_force)
        
        
                    # Create an integrator with a time step of 0.5 fs
                    temperature = 298.15 * kelvin
                    frictionCoeff = 1.0 / femtosecond
                    timeStep = 0.5 * femtosecond
                    integrator = LangevinMiddleIntegrator(temperature, frictionCoeff, timeStep)
        
                    # Create a simulation and set the initial positions and velocities
                    simulation = Simulation(topology, system, integrator)
                    simulation.context.setPositions(positions)
                    
                    start_time = time.perf_counter()
                    simulation.step(10)
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    
                    print(f"Execution time: {duration:.4f} seconds")
                    runtime.append(duration)
    
                # record in the df
                df[mol] = {model_id: runtime}
                df.to_csv("md_time.csv")

if __name__ == "__main__":
    main()