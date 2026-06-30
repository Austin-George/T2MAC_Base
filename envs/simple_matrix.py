import numpy as np

from .multiagentenv import MultiAgentEnv


class SimpleMatrixEnv(MultiAgentEnv):
    """Small cooperative matrix game for smoke-testing MARL algorithms."""

    def __init__(self, n_agents=2, n_actions=2, episode_limit=1, seed=None, **kwargs):
        del kwargs
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.episode_limit = episode_limit
        self._rng = np.random.default_rng(seed)
        self._t = 0

    def step(self, actions):
        self._t += 1
        actions = np.asarray(actions)
        reward = 1.0 if np.all(actions == actions[0]) else 0.0
        terminated = self._t >= self.episode_limit
        return reward, terminated, {"episode_limit": terminated}

    def get_obs(self):
        return [np.array([self._t / max(1, self.episode_limit)], dtype=np.float32) for _ in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        return 1

    def get_state(self):
        return np.array([self._t / max(1, self.episode_limit)], dtype=np.float32)

    def get_state_size(self):
        return 1

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        del agent_id
        return [1] * self.n_actions

    def get_total_actions(self):
        return self.n_actions

    def reset(self):
        self._t = 0
        return self.get_obs(), self.get_state()

    def render(self):
        return None

    def close(self):
        return None

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def save_replay(self):
        return None
