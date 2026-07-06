"""
T2MAC P2P Communication Learner
================================
This learner implements Q-learning with QMIX value decomposition for
training the T2MAC agents.

The training objective is standard 1-step TD-error with Double Q-learning:
    L = (Q_tot(s,a) - y)^2
    y = r + gamma * Q_tot_target(s', argmax_a Q_tot(s', a))

Q_tot is computed by the QMIX mixing network which combines individual
agent Q-values (evidence values) into a joint team value.

Note: The communication selector network has its OWN optimizer inside
the controller. This learner only trains the evidence encoder and mixer.

Reference: Rashid et al., "QMIX: Monotonic Value Function Factorisation
for Deep Multi-Agent Reinforcement Learning", ICML 2018.
"""
import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
import torch.nn.functional as F
from torch.distributions import Categorical


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        # Collect all trainable parameters: agent network + mixer network
        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        # === VALUE DECOMPOSITION MIXER ===
        # QMIX combines individual Q-values into Q_tot using state-dependent
        # hypernetworks with monotonicity constraints (dQ_tot/dQ_i >= 0).
        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()  # Q_tot = sum(Q_i) — simplest decomposition
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)  # Q_tot via monotonic hypernetwork
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)  # Target mixer for stable training

        # RMSprop optimizer for the main RL training
        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # Deep copy the entire MAC for the target network (Double Q-learning)
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """
        Main training step: compute TD-error loss and update all parameters.

        Args:
            batch:       EpisodeBatch of sampled episodes from the replay buffer
            t_env:       Current total environment steps (for logging)
            episode_num: Current episode number (for target network updates)
        """
        # --- Extract batch data ---
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        # Zero out the mask after termination (don't learn from padded data)
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # --- Forward pass: collect Q-values (evidence) for all timesteps ---
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # [batch, time, n_agents, n_actions]

        # --- Select Q-values for the actions that were actually taken ---
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # --- Target network forward pass ---
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # Skip first timestep for targets (we need Q(s_{t+1}))
        target_mac_out = th.stack(target_mac_out[1:], dim=1)

        # Mask unavailable actions in target with large negative value
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # --- Double Q-Learning ---
        # Use the LIVE network to SELECT the best action,
        # but use the TARGET network to EVALUATE its value.
        # This reduces overestimation bias.
        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # --- QMIX Mixing ---
        # Combine individual agent Q-values into team Q_tot using global state
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # --- TD Target ---
        # y = r + gamma * (1 - terminated) * max_a' Q_target(s', a')
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # --- TD Error Loss ---
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask

        # Mean squared TD error, averaged only over non-padded timesteps
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # --- Backpropagation ---
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # --- Periodic Target Network Update ---
        # Hard update: copy live network weights to target network
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # --- Logging ---
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        """Hard update: copy all live network weights to target networks."""
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        """Move all networks to GPU."""
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
