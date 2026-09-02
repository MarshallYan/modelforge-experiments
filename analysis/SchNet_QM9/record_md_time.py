from modelforge.potential.potential import load_inference_model_from_checkpoint

checkpoint_file_path = "/Users/syan/workdir/modelforge-experiments/experiments/exp14/downloads/checkpoints/model-vwgr40cg_v0/model.ckpt"
potential = load_inference_model_from_checkpoint(checkpoint_file_path, jit=False)


glucose = "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O"
phenylalanine = "c1ccc(cc1)C[C@H](C(=O)O)N"
caffeine = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
camptothecin = r"O=C\1N4\C(=C/C2=C/1COC(=O)[C@]2(O)CC)c3nc5c(cc3C4)cccc5"
nirmatrelvir = "CC1([C@@H]2[C@H]1[C@H](N(C2)C(=O)[C@H](C(C)(C)C)NC(=O)C(F)(F)F)C(=O)N[C@@H](C[C@@H]3CCNC3=O)C#N)C"


from modelforge.openmm.examples.helper_functions import openmm_topology_from_smiles

topology, positions = openmm_topology_from_smiles(smiles=nirmatrelvir, optimize=True)
atomic_numbers = [atom.element.atomic_number for atom in topology.atoms()]


from modelforge.openmm.potential import generate_compute
from openmm import PythonForce

comp = generate_compute(potential=potential, atomic_numbers=atomic_numbers)
system_force = PythonForce(comp)


import openmm
from openmm.unit import (
    kelvin,
    picosecond,
    femtosecond,
    nanometer,
    kilojoules_per_mole,
)
from openmm.app import PDBFile

# define the systme
system = openmm.System()
for atom in topology.atoms():
    system.addParticle(atom.element.mass)

# add the system_force instance defined above that wraps modelforge potential for PythonFroce
system.addForce(system_force)


import sys, time
from openmm import LangevinMiddleIntegrator
from openmm.app import Simulation, StateDataReporter, DCDReporter

# Create an integrator with a time step of 1 fs
temperature = 298.15 * kelvin
frictionCoeff = 1.0 / femtosecond
timeStep = 0.1 * femtosecond
integrator = LangevinMiddleIntegrator(temperature, frictionCoeff, timeStep)

# Create a simulation and set the initial positions and velocities
simulation = Simulation(topology, system, integrator)
simulation.context.setPositions(positions)


start_time = time.perf_counter()
simulation.step(10000)
end_time = time.perf_counter()

print(f"Execution time: {end_time - start_time:.4f} seconds")