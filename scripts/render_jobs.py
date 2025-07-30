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
):
    template = environment.get_template(template_path)
    return template.render()

def render_runtime(
        environment: Environment,
        template_path: str,
):
    template = environment.get_template(template_path)
    return template.render()

def render_training(
        environment: Environment,
        template_path: str,
):
    template = environment.get_template(template_path)
    return template.render()

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
    env = Environment(loader=FileSystemLoader("../default"))
    dataset_template = "datasets/qm9.jinja"
    potential_template = "potentials/aimnet2.jinja"
    runtime_template = "runtimes/runtime.jinja"
    training_template = "trainings/training.jinja"
    slurm_template = "scripts/slurm_job.jinja"

    assembled_config = (
        f"# ============================================================ #\n\n"
        f"{render_dataset(env, dataset_template)}"
        f"\n\n# ============================================================ #\n\n"
        f"{render_potential(env, potential_template)}"
        f"\n\n# ============================================================ #\n\n"
        f"{render_runtime(env, runtime_template)}"
        f"\n\n# ============================================================ #\n\n"
        f"{render_training(env, training_template)}"
        f"\n\n# ============================================================ #\n"
    )

    python_cmd = (
        f"python perform_training.py "
        f"--condensed_config_path config.toml --accelerator 'gpu' --device [0]"
    )

    slurm_script = render_slurm_job(env, slurm_template, job_name="test", python_cmd=python_cmd)

    # output
    config_path = f"./config.toml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w+") as f:
        f.write(assembled_config)

    run_locally_path = f"./run_locally.sh"
    with open(run_locally_path, "w+") as f:
        f.write(python_cmd)

    submit_slurm = f"./submit_slurm.sh"
    with open(submit_slurm, "w+") as f:
        f.write(slurm_script)
