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
        normalize: bool,
):
    template = environment.get_template(template_path)
    config = template.render(
        normalize=normalize,
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
    dataset_template = "qm9.jinja"
    potential_template = "schnet.jinja"
    runtime_template = "runtime.jinja"
    training_template = "training.jinja"
    slurm_template = "slurm_job.jinja"

    # rendering experiment configs
    range_seed = [42, 43, 44, 45, 46]
    options_normalize = ["true", "false"]

    count = 0
    for seed in range_seed:
        for normalize in options_normalize:

            # config names
            experiment_name = f"10(1)_{normalize}_({seed})"
            project = "per_atom_energy_normalization"
            group = "exp10_1"
            tags = [
                f"{seed=}",
                f"{normalize=}",
                "SchNet",
                "QM9",
            ]

            run_index = f"run{count:03d}"

            assembled_config = (
                f"# ============================================================ #\n\n"
                f"{render_dataset(env, dataset_template)}\n"
                f"\n\n# ============================================================ #\n\n"
                f"""{render_potential(
                    env,
                    potential_template,
                    normalize,
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

            local_python_cmd = (
                f"python ../../../../scripts/perform_training.py "
                f"--condensed_config_path config.toml "
                f"--accelerator 'cpu' --device 1"
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
                f.write(local_python_cmd)

            submit_slurm = f"runs/{run_index}/submit_slurm.sh"
            with open(submit_slurm, "w+") as f:
                f.write(slurm_script)

            # count runs
            count += 1
