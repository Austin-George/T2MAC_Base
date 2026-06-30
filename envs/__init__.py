import os
import sys
from functools import partial

from .custom_env import CustomEnv
from .simple_matrix import SimpleMatrixEnv

def env_fn(env, **kwargs):
    return env(**kwargs)


REGISTRY = {}
REGISTRY["custom"] = partial(env_fn, env=CustomEnv)
REGISTRY["simple_matrix"] = partial(env_fn, env=SimpleMatrixEnv)

try:
    from smac.env import StarCraft2Env

    REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
except ImportError:
    StarCraft2Env = None

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
