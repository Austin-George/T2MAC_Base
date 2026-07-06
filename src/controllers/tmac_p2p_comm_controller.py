"""
T2MAC P2P Communication Controller (Multi-Agent Controller - MAC)
=================================================================
This controller orchestrates the full T2MAC P2P communication pipeline.

It coordinates the flow between agents:
1. Observation encoding (forward pass through GRU)
2. Key-Value-Query generation (for targeted messaging)
3. Communication probability prediction (who to talk to)
4. Thresholding to binary send/no-send decisions
5. Evidence aggregation with DST combination
6. Training the communication selector network (separate optimizer)

Key Design: All agents SHARE the same neural network parameters.
This is the standard "parameter sharing" approach in MARL. Each agent
gets differentiated behavior through its unique observation and agent ID.

Reference: Paper Section 4.2, "Selective Engagement"
"""
import torch
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import itertools


class VffacMAC:
    """Value Function Factorization with Attention Communication - Multi-Agent Controller."""

    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        # Action selection strategy (epsilon-greedy for Q-learning)
        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

        # === SEPARATE OPTIMIZER for the communication selector network ===
        # Paper Section 4.2: The selector network (msg_adapters + fc_msg_2) has its
        # OWN Adam optimizer, independent of the main RL RMSprop optimizer.
        # This allows the communication learning to proceed at its own pace.
        # FIX: fc_msg_1 was replaced by msg_adapters (3 heterogeneous adapters).
        param = itertools.chain(self.agent.msg_adapters.parameters(), self.agent.fc_msg_2.parameters())
        self.msg_optim = torch.optim.Adam(param, lr=0.001)

        # FIX: CrossEntropyLoss doesn't need to be on a specific device.
        # Removed .cuda() call for CPU compatibility.
        self.crit_fun = torch.nn.CrossEntropyLoss()

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        """Select actions for all agents using epsilon-greedy over evidence values."""
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)
        return chosen_actions

    @torch.no_grad()
    def generate_send_target(self, send_prob):
        """
        Paper Section 4.2: Convert soft probabilities to binary send decisions.

        Threshold = 0.75: If the selector network predicts > 75% chance that
        a message is valuable, the agent will send it.

        Args:
            send_prob: [batch, n_agents, n_agents] - soft probabilities

        Returns:
            send_target: [batch, n_agents, n_agents] - binary decisions (0 or 1)
        """
        send_target = torch.where(send_prob > 0.75, 1, 0)
        if len(send_target.shape) == 2:
            send_target = send_target.unsqueeze(0)
        return send_target

    def forward(self, ep_batch, t, test_mode=False, counterfactual=False):
        """
        Full T2MAC forward pass for one timestep.

        Steps:
        1. Build observation inputs
        2. Update GRU hidden states
        3. Generate Key/Value/Query for attention messaging
        4. Predict communication probabilities
        5. Threshold into binary send targets (subtract identity to prevent self-comm)
        6. Aggregate messages with DST evidence combination
        7. Train the selector network using uncertainty-based pseudo-labels
        """
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        # Step 1: Update hidden states through GRU
        self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # (Optional) Counterfactual: compute Q-values without communication for comparison
        if counterfactual:
            q_without_comm = self.agent.q_without_communication(self.hidden_states)

        # Step 2: Generate communication vectors
        agents_key, agents_value, agents_query = self.agent.communicate(self.hidden_states)

        # Step 3: Predict who to communicate with
        send_prob = self.agent.generate_send_prob(agent_inputs)

        # Step 4: Threshold and remove self-communication
        # FIX: Use device of send_prob instead of hardcoded .cuda()
        device = send_prob.device
        send_target = torch.clamp(
            self.generate_send_target(send_prob) - torch.eye(self.n_agents, device=device),
            0, 1
        ).int().detach()

        # Step 5: Aggregate evidence with DST combination
        agents_out, u_err = self.agent.aggregate(
            agents_query, agents_key, agents_value,
            self.hidden_states, send_target
        )  # agents_out: [batch, n_agents, n_actions], u_err: uncertainty change

        # Step 6: Train the selector network (ONLY during training batches)
        # Paper Section 4.2, Eq. 4: Binary cross-entropy loss with pseudo-labels.
        # Pseudo-labels come from the actual uncertainty reduction observed.
        if u_err.shape[0] == self.args.batch_size:
            true_label = self.generate_true_label(u_err, send_target)
            loss = self.crit_fun(true_label, send_prob)
            self.msg_optim.zero_grad()
            loss.backward()
            self.msg_optim.step()

        return (agents_out, q_without_comm) if counterfactual else agents_out

    def generate_true_label(self, u_err, send_target):
        """
        Paper Section 4.2: Generate pseudo-labels for training the selector network.

        Uses uncertainty change (u_err) as a signal:
        - If communication changed uncertainty by more than min_uncer threshold,
          label it as 1 (communication was valuable)
        - Otherwise label it as 0 (communication was not useful)

        Args:
            u_err:       [batch, n_agents, 1] - uncertainty change per agent
            send_target: [batch, n_agents, n_agents] - who sent to whom

        Returns:
            true_label: [batch, n_agents, n_agents] - binary pseudo-labels
        """
        u_err = u_err.repeat(1, 1, send_target.shape[-1])  # Broadcast to match send_target
        true_label = torch.where(u_err > self.args.min_uncer, 1, 0)
        return true_label.float()

    def init_hidden(self, batch_size):
        """Initialize GRU hidden states to zeros for all agents."""
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def parameters(self):
        """Return all agent parameters (used by the learner's optimizer)."""
        return self.agent.parameters()

    def load_state(self, other_mac):
        """Copy weights from another MAC (used for target network updates)."""
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        """Move agent to GPU."""
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        """Create the shared agent network from the registry."""
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        """
        Build the input tensor for all agents at timestep t.

        The input is a concatenation of:
        1. Local observation o_i^t
        2. Previous action a_i^{t-1} (one-hot encoded)
        3. Agent ID (one-hot encoded) — this enables heterogeneous behavior
           even with shared parameters

        Returns:
            inputs: [batch_size * n_agents, input_dim]
        """
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # Local observation
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))  # No prev action at t=0
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])  # Previous action (one-hot)
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
            # One-hot agent ID: enables each agent to learn specialized behavior

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        """Calculate total input dimension: obs_size + action_size + n_agents."""
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
