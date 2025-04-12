import os
from jinja2 import Environment, FileSystemLoader, Template


def render_dataset(
        environment: Environment,
        template_path: str,
        version_select: str,
):
    template = environment.get_template(template_path)
    config = template.render(version_select=version_select)
    return config

def render_potential(
        environment: Environment,
        template_path: str,
        predicted_properties: list,
        predicted_dim: list,
        properties_to_featurize: list,
        properties_to_process: list,
        postprocessing: str,
):
    template = environment.get_template(template_path)
    config = template.render(
        predicted_properties = predicted_properties,
        predicted_dim = predicted_dim,
        properties_to_featurize = properties_to_featurize,
        properties_to_process = properties_to_process,
        postprocessing = postprocessing,
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
        loss_components: list,
        loss_weight: str,
):
    template = environment.get_template(template_path)
    config = template.render(
        project=project,
        group=group,
        tags=tags,
        seed=seed,
        loss_components = loss_components,
        loss_weight = loss_weight,
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
    dataset_template = "tmqm_xtb.jinja"
    potential_template = "schnet.jinja"
    runtime_template = "runtime.jinja"
    training_template = "training.jinja"
    slurm_template = "slurm_job.jinja"

    # rendering experiment configs
    range_seed = [42, 43, 44, 45, 46]
    options_version_select = ["PdZnFeCu_T100K_single_config_v1", "PdZnFeCu_T100K_v1.1"]
    options_loss_components = [
        ["per_atom_energy"],
        ["per_atom_energy", "per_atom_force"],
        ["per_atom_energy", "per_atom_force", "per_system_dipole_moment"],
    ]

    count = 0
    for seed in range_seed:
        for version_select in options_version_select:
            for loss_components in options_loss_components:
                if len(loss_components) == 1:   # E
                    experiment_name = "E_"
                    tags = ["E"]

                    predicted_properties = loss_components
                    predicted_dim = [1]
                    properties_to_featurize = ["atomic_number"]
                    properties_to_process = ["per_atom_energy"]
                    postprocessing = ""

                    loss_weight = """
per_atom_energy = 1
                    """

                elif len(loss_components) == 2: # E, F
                    experiment_name = "EF_"
                    tags = ["E", "F"]

                    predicted_properties = loss_components
                    predicted_dim = [1, 1]
                    properties_to_featurize = ["atomic_number"]
                    properties_to_process = ["per_atom_energy"]
                    postprocessing = ""

                    loss_weight = """
per_atom_energy = 1
per_atom_force = 0.01
                    """

                elif len(loss_components) == 3: # E, F, mu
                    experiment_name = "EFmu_"
                    tags = ["E", "F", "mu"]

                    predicted_properties = ["per_atom_energy", "per_atom_charge", "per_system_total_charge"]
                    predicted_dim = [1, 1, 1]
                    properties_to_featurize = ["atomic_number", "per_system_total_charge"]
                    properties_to_process = ["per_atom_energy", "per_atom_charge"]
                    postprocessing = """
[potential.postprocessing_parameter.per_atom_charge]
conserve = true
conserve_strategy = "default" 
                    """

                    loss_weight = """
per_atom_energy = 1
per_atom_force = 0.01
per_system_dipole_moment = 0.01
                    """
                
                # config
                experiment_name += (
                    f"{version_select}"
                    f"({seed})"
                )
                project = "schnet_tmqm_xtb"
                group = version_select
                run_index = f"run{count:03d}"

                assembled_config = (
                    f"# ============================================================ #\n\n"
                    f"{render_dataset(env, dataset_template, version_select)}"
                    f"\n\n# ============================================================ #\n\n"
                    f"""{render_potential(
                        env,
                        potential_template,
                        predicted_properties,
                        predicted_dim,
                        properties_to_featurize,
                        properties_to_process,
                        postprocessing,
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
                        loss_weight,
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
