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
        number_of_interaction_modules: int,
        number_of_filters: int,
        number_of_per_atom_features: int,
):
    template = environment.get_template(template_path)
    config = template.render(
        number_of_interaction_modules=number_of_interaction_modules,
        number_of_filters=number_of_filters,
        number_of_per_atom_features=number_of_per_atom_features,
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
        seed: int,
):
    template = environment.get_template(template_path)
    config = template.render(seed=seed)
    return config


def render_slurm_job(
        environment: Environment,
        template_path: str,
        job_name: str,
        python_cmd: str,
):
    template = environment.get_template(template_path)
    slurm_script = template.render(job_name=job_name, python_cmd=python_cmd)
    return slurm_script


if __name__ == "__main__":
    # input
    env = Environment(loader=FileSystemLoader("templates"))
    dataset_template = "qm9.jinja"
    potential_template = "schnet.jinja"
    runtime_template = "runtime.jinja"
    training_template = "training.jinja"
    slurm_template = "slurm_job.jinja"

    # rendering experiment configs
    range_seed = [42, 43, 44, 45, 46]
    range_number_of_interaction_modules = [3, 6, 9, 12]
    range_number_of_filters = [32, 64, 128, 256]
    range_number_of_per_atom_features = [16, 32, 64, 128, 256, 512, 1024]

    count = 0
    for seed in range_seed:
        for number_of_interaction_modules in range_number_of_interaction_modules:
            for number_of_filters in range_number_of_filters:
                for number_of_per_atom_features in range_number_of_per_atom_features:
                    experiment_name = (f"{number_of_interaction_modules}"
                                       +f"_{number_of_filters}"
                                       +f"_{number_of_per_atom_features}")

                    assembled_config = (
                          "# ============================================================ #\n\n"
                        + render_dataset(env, dataset_template)
                        + "\n\n# ============================================================ #\n\n"
                        + render_potential(
                            env,
                            potential_template,
                            number_of_interaction_modules,
                            number_of_filters,
                            number_of_per_atom_features
                        )
                        + "\n\n# ============================================================ #\n\n"
                        + render_runtime(env, runtime_template, experiment_name)
                        + "\n\n# ============================================================ #\n\n"
                        + render_training(env, training_template, seed)
                        + "\n\n# ============================================================ #\n"
                    )

                    python_cmd = (
                        f"python perform_training.py "
                        f"--condensed_config_path config.toml --accelerator 'gpu' --device [0]"
                    )

                    slurm_script = render_slurm_job(env, slurm_template, job_name="test", python_cmd=python_cmd)

                    # output
                    config_path = f"run{count:02d}/config.toml"
                    os.makedirs(os.path.dirname(config_path), exist_ok=True)

                    with open(config_path, "w+") as f:
                        f.write(assembled_config)

                    run_locally_path = f"run{count:02d}/run_locally.sh"
                    with open(run_locally_path, "w+") as f:
                        f.write(python_cmd)

                    submit_slurm = f"run{count:02d}/submit_slurm.sh"
                    with open(submit_slurm, "w+") as f:
                        f.write(slurm_script)

                    # count runs
                    count += 1
