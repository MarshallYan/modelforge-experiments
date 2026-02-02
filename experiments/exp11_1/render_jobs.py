import os
from jinja2 import Environment, FileSystemLoader, Template


def render_dataset(
        environment: Environment,
        template_path: str,
):
    template = environment.get_template(template_path)
    return template.render()

def render_potential(
        environment: Environment,
        template_path: str,
        contributions: list,
):
    template = environment.get_template(template_path)
    config = template.render(
        contributions=contributions,
    )
    return config

def render_runtime(
        environment: Environment,
        template_path: str,
        experiment_name: str,
):
    template = environment.get_template(template_path)
    config = template.render(experiment_name=experiment_name)
    return config

def render_training(
        environment: Environment,
        template_path: str,
        project: str,
        group: str,
        tags: list,
        seed: int,
        per_system_energy: float,
        per_system_dipole_moment: float,
        per_atom_charge: float,
        per_atom_force: float,
):
    template = environment.get_template(template_path)
    config = template.render(
        project=project,
        group=group,
        tags=tags,
        seed=seed,
        per_system_energy=per_system_energy,
        per_atom_charge = per_atom_charge,
        per_system_dipole_moment = per_system_dipole_moment,
        per_atom_force = per_atom_force,
    )
    return config


def render_slurm_job(
        environment: Environment,
        template_path: str,
        job_name: str,
        python_cmd: str,
        run_index: str,
):
    template = environment.get_template(template_path)
    slurm_script = template.render(
        job_name=job_name,
        python_cmd=python_cmd,
        run_index=run_index,
    )
    return slurm_script


if __name__ == "__main__":
    # input
    env = Environment(loader=FileSystemLoader("templates"))
    dataset_template = "qm9.jinja"
    potential_template = "aimnet2.jinja"
    runtime_template = "runtime.jinja"
    training_template = "training.jinja"
    slurm_template = "slurm_job.jinja"

    # rendering experiment configs
    range_seed = [42, 43, 44, 45, 46]
    options_contributions = [
        [],
        ['per_system_vdw_energy'],
        ['per_system_electrostatic_energy', 'per_system_vdw_energy'],
    ]
    options_per_system_energy = [0.01]
    options_per_system_dipole_moment = [0, 1]
    options_per_atom_charge = [0, 0.1]
    options_per_atom_force = [0, 0.001]

    count = 0
    for seed in range_seed:
        for contributions in options_contributions:
            for per_system_energy in options_per_system_energy:
                for per_system_dipole_moment in options_per_system_dipole_moment:
                    for per_atom_charge in options_per_atom_charge:
                        for per_atom_force in options_per_atom_force:

                            # config names
                            experiment_name = f"11(1)_E{len(contributions)}_{per_system_energy}_{per_system_dipole_moment}_{per_atom_charge}_{per_atom_force}({seed})"
                            project = "aimnet2_qm9"
                            group = "exp11_1"
                            tags = [
                                f"{seed=}",
                                f"contributions={len(contributions)}",
                                f"{per_system_energy=}",
                                f"{per_system_dipole_moment=}",
                                f"{per_atom_charge=}",
                                f"{per_atom_force=}",
                            ]

                            run_index = f"run{count:03d}"

                            assembled_config = (
                                f"# ============================================================ #\n\n"
                                f"{render_dataset(env, dataset_template)}\n"
                                f"\n\n# ============================================================ #\n\n"
                                f"""{render_potential(
                                    env,
                                    potential_template,
                                    contributions,
                                )}"""
                                f"\n\n# ============================================================ #\n\n"
                                f"{render_runtime(env, runtime_template, experiment_name)}"
                                f"\n\n# ============================================================ #\n\n"
                                f"""{render_training(
                                    env, 
                                    training_template, 
                                    project,
                                    group,
                                    tags,
                                    seed,
                                    per_system_energy,
                                    per_system_dipole_moment,
                                    per_atom_charge,
                                    per_atom_force,
                                )}"""
                                f"\n\n# ============================================================ #\n"
                            )

                            python_cmd = (
                                f"python ../../../../scripts/perform_training.py "
                                f"--condensed_config_path config.toml "
                                f"--accelerator 'gpu' --device [0]"
                            )

                            slurm_script = render_slurm_job(
                                env,
                                slurm_template,
                                job_name=run_index,
                                python_cmd=python_cmd,
                                run_index=run_index,
                            )

                            # output
                            config_path = f"runs/{run_index}/config.toml"
                            os.makedirs(os.path.dirname(config_path), exist_ok=True)

                            with open(config_path, "w+") as f:
                                f.write(assembled_config)

                            run_locally_path = f"runs/{run_index}/run_locally.sh"
                            with open(run_locally_path, "w+") as f:
                                f.write(python_cmd)

                            submit_slurm = f"runs/{run_index}/submit_slurm.sh"
                            with open(submit_slurm, "w+") as f:
                                f.write(slurm_script)

                            # count runs
                            count += 1
