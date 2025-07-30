import os
from jinja2 import Environment, FileSystemLoader, Template


def render_dataset(
        environment: Environment,
        template_path: str,
        version_select: str,
):
    template = environment.get_template(template_path)
    return template.render(version_select=version_select)

def render_potential(
        environment: Environment,
        template_path: str,
):
    template = environment.get_template(template_path)
    config = template.render()
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
        loss_components: list,
):
    template = environment.get_template(template_path)
    config = template.render(
        project=project,
        group=group,
        tags=tags,
        seed=seed,
        per_system_energy=loss_components[0],
        per_system_dipole_moment=loss_components[1],
        per_atom_charge=loss_components[2],
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
    dataset_template = "tmqm_openff.jinja"
    potential_template = "aimnet2.jinja"
    runtime_template = "runtime.jinja"
    training_template = "training.jinja"
    slurm_template = "slurm_job.jinja"

    # rendering experiment configs
    range_seed = [42, 43, 44, 45, 46]
    options_version_select = [
        "full_dataset_sm1_v1.1",
        "full_dataset_sm3_v1.1",
        "full_dataset_sm5_v1.1",
        "full_dataset_v1.1",
    ]
    option_loss_components = [
        [0.0001, 0.1, 1],
        [0.0001, 0, 1],
        [0.0001, 0.1, 0],
        [1, 0, 0],
    ]


    count = 0
    for seed in range_seed:
        for version_select in options_version_select:
            for loss_components in option_loss_components:
                # assign loss weights
                per_system_energy, per_system_dipole_moment, per_atom_charge = loss_components

                # config names
                experiment_name = f"{version_select}_{loss_components}_({seed})"
                project = "aimnet2_tmqm_openff"
                group = version_select
                tags = [
                    f"seed={seed}",
                    f"{version_select}",
                    f"per_system_energy={per_system_energy}",
                    f"per_system_dipole_moment={per_system_dipole_moment}",
                    f"per_atom_charge={per_atom_charge}",
                ]

                run_index = f"run{count:03d}"

                assembled_config = (
                    f"# ============================================================ #\n\n"
                    f"{render_dataset(env, dataset_template, version_select)}\n"
                    f"\n\n# ============================================================ #\n\n"
                    f"""{render_potential(
                        env,
                        potential_template,
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
                        loss_components,
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
