import os
from jinja2 import Environment, FileSystemLoader, Template


def render_dataset(
        environment: Environment,
        template_path: str,
):
    template = environment.get_template(template_path)
    config = template.render()
    return config

def render_potential(
        environment: Environment,
        template_path: str,
        maximum_interaction_radius: float
):
    template = environment.get_template(template_path)
    config = template.render(
        maximum_interaction_radius=maximum_interaction_radius
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
    dataset_template = "tmqm_xtb.jinja"
    potential_template = "schnet.jinja"
    runtime_template = "runtime.jinja"
    training_template = "training.jinja"
    slurm_template = "slurm_job.jinja"

    # rendering experiment configs
    range_seed = [42, 43, 44, 45, 46]

    range_maximum_interaction_radius = [6, 7, 8, 9, 10]

    count = 0
    for seed in range_seed:
        for maximum_interaction_radius in range_maximum_interaction_radius:
            # config
            experiment_name = f"maximum_interaction_radius_{maximum_interaction_radius}({seed})"
            project = "schnet_tmqm_xtb"
            group = f"PdZnFeCu_T100K_v1.1"
            tags = [f"maximum_interaction_radius: {maximum_interaction_radius}", "E", "F", "mu"]
            run_index = f"run{count:03d}"

            assembled_config = (
                f"# ============================================================ #\n\n"
                f"{render_dataset(env, dataset_template)}"
                f"\n\n# ============================================================ #\n\n"
                f"""{render_potential(
                    env,
                    potential_template,
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
