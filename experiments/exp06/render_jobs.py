import os
from jinja2 import Environment, FileSystemLoader, Template


def render_dataset(
        environment: Environment,
        template_path: str,
        element_filter: list,
):
    template = environment.get_template(template_path)
    return template.render(element_filter=element_filter)

def render_potential(
        environment: Environment,
        template_path: str,
        number_of_interaction_modules: int,
        number_of_filters: int,
        number_of_per_atom_features: int,
        maximum_interaction_radius: float,
):
    template = environment.get_template(template_path)
    config = template.render(
        number_of_interaction_modules=number_of_interaction_modules,
        number_of_filters=number_of_filters,
        number_of_per_atom_features=number_of_per_atom_features,
        maximum_interaction_radius=maximum_interaction_radius,
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
):
    template = environment.get_template(template_path)
    config = template.render(
        project=project,
        group=group,
        tags=tags,
        seed=seed,
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
    dataset_template = "tmqm.jinja"
    potential_template = "schnet.jinja"
    runtime_template = "runtime.jinja"
    training_template = "training.jinja"
    slurm_template = "slurm_job.jinja"

    # rendering experiment configs
    range_seed = [42, 43, 44, 45, 46]
    range_number_of_interaction_modules = [6]
    range_number_of_filters = [128, 256]
    range_number_of_per_atom_features = [128, 256]
    range_maximum_interaction_radius = [5, 6, 7]
    options_element_filter = [
        [[26]],   # Fe
        [[29], [30], [46]],   # Cu, Zn, Pd
        [], # all
    ]

    count = 0
    for seed in range_seed:
        for number_of_interaction_modules in range_number_of_interaction_modules:
            for number_of_filters in range_number_of_filters:
                for number_of_per_atom_features in range_number_of_per_atom_features:
                    for maximum_interaction_radius in range_maximum_interaction_radius: 
                        for element_filter in options_element_filter:

                            # config names
                            experiment_name = (
                                f"_{number_of_filters}"
                                f"_{number_of_per_atom_features}"
                                f"_{maximum_interaction_radius}"
                                f"({seed})"
                            )
                            project = "schnet_tmqm_FevsCuZnPd"
                            group = f"seed: {seed}"
                            tags = [
                                f"number_of_filters: {number_of_filters}",
                                f"number_of_per_atom_features: {number_of_per_atom_features}",
                                f"maximum_interaction_radius: {maximum_interaction_radius}",
                                f"element_filter: {element_filter}",
                                "iris",
                            ]
                            run_index = f"run{count:03d}"

                            assembled_config = (
                                f"# ============================================================ #\n\n"
                                f"{render_dataset(env, dataset_template, element_filter)}"
                                f"\n\n# ============================================================ #\n\n"
                                f"""{render_potential(
                                    env,
                                    potential_template,
                                    number_of_interaction_modules,
                                    number_of_filters,
                                    number_of_per_atom_features,
                                    maximum_interaction_radius,
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