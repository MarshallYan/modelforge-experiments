import toml
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


if __name__ == "__main__":
    # input
    env = Environment(loader=FileSystemLoader("../default"))
    dataset_template = "datasets/qm9.jinja"
    potential_template = "potentials/schnet.jinja"
    runtime_template = "runtimes/runtime.jinja"
    training_template = "trainings/training.jinja"

    config = (
          "# ============================================================ #\n\n"
        + render_dataset(env, dataset_template)
        + "\n\n# ============================================================ #\n\n"
        + render_potential(env, potential_template)
        + "\n\n# ============================================================ #\n\n"
        + render_runtime(env, runtime_template)
        + "\n\n# ============================================================ #\n\n"
        + render_training(env, training_template)
        + "\n\n# ============================================================ #\n"
    )


    # output
    print(config)
    config_path = f"./config.toml"
    with open(config_path, "w+") as f:
        f.write(config)


