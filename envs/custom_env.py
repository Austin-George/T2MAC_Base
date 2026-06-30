import importlib


class CustomEnv:
    """Load a user-provided MultiAgentEnv-compatible class from config."""

    def __init__(self, class_path, **kwargs):
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        env_cls = getattr(module, class_name)
        self.env = env_cls(**kwargs)

    def __getattr__(self, name):
        return getattr(self.env, name)
