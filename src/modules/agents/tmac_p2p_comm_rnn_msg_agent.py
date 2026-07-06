"""
T2MAC P2P Communication Agent
=================================
This is the core neural network for the T2MAC (Targeted and Trusted Multi-Agent
Communication) framework from the AAAI 2024 paper.

This agent implements three key innovations:
1. Attention-based Targeted Messaging (Section 4, "Targeted Messaging")
2. Evidential Deep Learning via Dirichlet distributions (Section 4.1, "Theory of Evidence")
3. Dempster-Shafer Theory for evidence combination (Section 4.3, "Evidence-Driven Integration")
4. Communication partner selection (Section 4.2, "Selective Engagement")

The agent takes observations, produces Q-values as "evidence" for a Dirichlet
distribution, and uses Dempster-Shafer Theory to combine evidence from multiple
agents in a mathematically principled way.

Reference: Sun et al., "T2MAC: Targeted and Trusted Multi-Agent Communication
through Selective Engagement and Evidence-Driven Integration", AAAI 2024.
"""
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F


class RnnMsgAgent(nn.Module):
    """
    The T2MAC P2P Communication Agent with Heterogeneous Adapters.

    Architecture:
        Input [obs, last_action, agent_id] -> adapters (per-type Linear+ReLU) -> GRU -> hidden state
        hidden state -> Key, Value, Query projections (for attention communication)
        hidden state -> msg_adapters + fc_msg_2 (for communication partner selection)
        Attention aggregation + DST combination -> Evidence (used as Q-values)

    Input shape:  [batch_size * n_agents, input_dim]
    Output shape: [batch_size, n_agents, n_actions]
    Hidden state: [batch_size, n_agents, rnn_hidden_dim]
    """

    def __init__(self, input_dim, args):
        super().__init__()

        self.args = args

        # ========== HETEROGENEOUS ADAPTERS ==========
        # We replace the single fc1 with a dictionary of 3 adapters!
        # The input_dim passed here is the MAX size (160 + actions + agent_id)
        # We calculate the base extra features by subtracting the max obs shape (160)
        base_extra = input_dim - 160  
        
        self.adapters = nn.ModuleDict({
            "medivac": nn.Linear(110 + base_extra, args.rnn_hidden_dim),
            "marauder": nn.Linear(130 + base_extra, args.rnn_hidden_dim),
            "marine": nn.Linear(160 + base_extra, args.rnn_hidden_dim)
        })

        # ========== RECURRENT MEMORY ==========
        # GRU cell maintains temporal memory across timesteps.
        # This lets each agent remember what happened in previous frames.
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        # ========== Q-VALUE / EVIDENCE HEAD ==========
        # Outputs one value per action. In T2MAC, these Q-values are
        # reinterpreted as "evidence" for a Dirichlet distribution.
        # (Paper Section 4.1, Equation 2: e_k >= 0)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        # ========== ATTENTION COMMUNICATION LAYERS ==========
        # These implement the Key-Query-Value attention mechanism for
        # targeted messaging (similar to TarMAC, Das et al. 2019).
        # Each agent projects its hidden state into Key, Value, and Query vectors.
        self.fc_value = nn.Linear(args.rnn_hidden_dim, args.n_value)   # Value: what information to share
        self.fc_key = nn.Linear(args.rnn_hidden_dim, args.n_key)       # Key: what information this agent has
        self.fc_query = nn.Linear(args.rnn_hidden_dim, args.n_query)   # Query: what information this agent needs

        # Attention weight computation: takes [query, all_keys] and outputs
        # a weight for each agent, determining how much to listen to each.
        self.fc_attn = nn.Linear(args.n_query + args.n_key * args.n_agents, args.n_agents)

        # Combines the attention output with the agent's own hidden state
        # (shortcut/residual connection to preserve self-information).
        self.fc_attn_combine = nn.Linear(args.n_value + args.rnn_hidden_dim, args.rnn_hidden_dim)

        # ========== COMMUNICATION SELECTOR NETWORK ==========
        # Paper Section 4.2: "Selective Engagement"
        # This 2-layer MLP learns WHEN and WITH WHOM to communicate.
        # It outputs a probability distribution over all agents, indicating
        # the likelihood of sending a message to each teammate.
        # trained with its own optimizer in the controller.
        self.msg_adapters = nn.ModuleDict({
            "medivac": nn.Linear(110 + base_extra, args.rnn_hidden_dim),
            "marauder": nn.Linear(130 + base_extra, args.rnn_hidden_dim),
            "marine": nn.Linear(160 + base_extra, args.rnn_hidden_dim)
        })
        self.fc_msg_2 = nn.Linear(args.rnn_hidden_dim, args.n_agents)  # Output: prob per agent

    def forward(self, x, hidden):
        """
        Forward pass: encode observation and update hidden state.

        This does NOT produce Q-values yet. Q-values come from aggregate()
        after the communication phase.

        Args:
            x:      [batch_size * n_agents, input_dim] - raw observations
            hidden: [batch_size, n_agents, rnn_hidden_dim] - previous hidden state

        Returns:
            h_out: [batch_size, n_agents, rnn_hidden_dim] - updated hidden state
        """
        # --- HETEROGENEOUS ROUTING ---
        batch_size = x.shape[0] // self.args.n_agents
        x_reshaped = x.view(batch_size, self.args.n_agents, -1)
        
        adapted_x_list = []
        for i in range(self.args.n_agents):
            agent_x = x_reshaped[:, i, :]
            extra = agent_x[:, 160:]  # Extra info (last action, agent id)
            
            if i == 0: # Medivac
                valid_obs = agent_x[:, :110]
                adapter_input = torch.cat([valid_obs, extra], dim=1)
                adapted = F.relu(self.adapters["medivac"](adapter_input))
            elif i in [1, 2]: # Marauder
                valid_obs = agent_x[:, :130]
                adapter_input = torch.cat([valid_obs, extra], dim=1)
                adapted = F.relu(self.adapters["marauder"](adapter_input))
            else: # Marine
                valid_obs = agent_x[:, :160]
                adapter_input = torch.cat([valid_obs, extra], dim=1)
                adapted = F.relu(self.adapters["marine"](adapter_input))
                
            adapted_x_list.append(adapted)
            
        x = torch.stack(adapted_x_list, dim=1).view(-1, self.args.rnn_hidden_dim)
        # -----------------------------
        h_in = hidden.view(-1, self.args.rnn_hidden_dim)  # Flatten for GRU
        h_out = self.rnn(x, h_in)  # Update hidden state with GRU
        h_out = h_out.view(-1, self.args.n_agents, self.args.rnn_hidden_dim)  # Reshape back
        return h_out

    def generate_send_prob(self, obs):
        """
        Paper Section 4.2: Selective Engagement - Communication Selector Network.

        Given the current observation, predict the probability of sending
        a message to each other agent.

        The output p_ij represents: "How likely is agent i to benefit agent j
        by sending a message?"

        These probabilities are later thresholded (> 0.75) in the controller
        to produce binary send/no-send decisions.

        Args:
            obs: [batch_size * n_agents, input_dim] - raw observations

        Returns:
            s: [batch_size, n_agents, n_agents] - send probabilities p_ij
        """
        batch = int(obs.shape[0] / self.args.n_agents)
        
        # --- HETEROGENEOUS ROUTING FOR COMMUNICATION ---
        obs_reshaped = obs.view(batch, self.args.n_agents, -1)
        
        f_list = []
        for i in range(self.args.n_agents):
            agent_obs = obs_reshaped[:, i, :]
            extra = agent_obs[:, 160:]
            
            if i == 0:
                valid_obs = agent_obs[:, :110]
                adapter_input = torch.cat([valid_obs, extra], dim=1)
                f_i = self.msg_adapters["medivac"](adapter_input)
            elif i in [1, 2]:
                valid_obs = agent_obs[:, :130]
                adapter_input = torch.cat([valid_obs, extra], dim=1)
                f_i = self.msg_adapters["marauder"](adapter_input)
            else:
                valid_obs = agent_obs[:, :160]
                adapter_input = torch.cat([valid_obs, extra], dim=1)
                f_i = self.msg_adapters["marine"](adapter_input)
                
            f_list.append(f_i)
            
        f = torch.stack(f_list, dim=1).view(-1, self.args.rnn_hidden_dim)
        # -----------------------------------------------
        
        s = self.fc_msg_2(f)    # Project to n_agents dimensions
        # Softmax normalizes across receivers (probabilities sum to 1 per sender)
        s = nn.Softmax(dim=-1)(s).reshape(batch, self.args.n_agents, s.shape[-1])
        return s

    def q_without_communication(self, h_out):
        """
        Compute Q-values WITHOUT any communication (for ablation/baseline comparison).
        Simply passes the hidden state through the final linear layer.
        """
        q_without_comm = self.fc2(h_out)
        return q_without_comm

    def communicate(self, hidden):
        """
        Generate Key, Value, and Query vectors for attention-based communication.

        Each agent projects its hidden state into three vectors:
        - Key:   "Here is what I know" (used by others to compute attention weights)
        - Value: "Here is the information I can share" (the actual message content)
        - Query: "Here is what I need to know" (used to attend to relevant agents)

        Args:
            hidden: [batch_size, n_agents, rnn_hidden_dim]

        Returns:
            key:   [batch_size, n_agents, n_key]
            value: [batch_size, n_agents, n_value]
            query: [batch_size, n_agents, n_query]
        """
        key = self.fc_key(hidden)
        value = self.fc_value(hidden)
        query = self.fc_query(hidden)
        return key, value, query

    def aggregate(self, query, key, value, hidden, send_target):
        """
        The main communication + evidence extraction pipeline.

        This method:
        1. Computes attention-weighted aggregation of all agents' values
        2. Extracts evidence (non-negative Q-values) from the aggregated output
        3. Computes pre-communication uncertainty (Paper Eq. 2: u = K/S)
        4. Combines evidence from selected senders using DST (Paper Eq. 5-7)
        5. Computes post-communication uncertainty
        6. Returns the combined evidence and the uncertainty change

        Args:
            query:       [batch_size, n_agents, n_query]
            key:         [batch_size, n_agents, n_key]
            value:       [batch_size, n_agents, n_value]
            hidden:      [batch_size, n_agents, rnn_hidden_dim]
            send_target: [batch_size, n_agents, n_agents] - binary communication mask

        Returns:
            evidence: [batch_size, n_agents, n_actions] - combined evidence (used as Q-values)
            u_delta:  [batch_size, n_agents, 1] - uncertainty change (cmb_u - ori_u)
        """
        n_agents = self.args.n_agents

        # --- Step 1: Attention-based message aggregation ---
        # Broadcast all agents' keys to every agent
        _key = torch.cat([key[:, i, :] for i in range(n_agents)], dim=-1).unsqueeze(1).repeat(1, n_agents, 1)
        query_key = torch.cat([query, _key], dim=-1)  # [batch, n_agents, n_query + n_agents*n_key]

        # Compute attention weights: how much should each agent listen to each other agent?
        attn_weights = F.softmax(self.fc_attn(query_key), dim=-1)  # [batch, n_agents, n_agents]

        # Weighted sum of values (the actual message aggregation)
        attn_applied = torch.bmm(attn_weights, value)  # [batch, n_agents, n_value]

        # Shortcut connection: combine attention output with agent's own hidden state.
        # This preserves the agent's own information alongside received messages.
        attn_combined = torch.cat([attn_applied, hidden], dim=-1)
        attn_combined = F.relu(self.fc_attn_combine(attn_combined))

        # --- Step 2: Evidence extraction ---
        # Paper Section 4.1, Eq. 2: Evidence must be non-negative (e_k >= 0).
        # We use ReLU (implemented as clamp) to ensure this.
        q = self.fc2(attn_combined)  # [batch, n_agents, n_actions]
        evidence = torch.clamp(q, 0, torch.inf)  # e_k = ReLU(q) -- non-negative evidence

        # --- Step 3: Pre-communication uncertainty ---
        # Paper Eq. 2: alpha_k = e_k + 1, S = sum(alpha_k), u = K / S
        alpha = evidence + 1                             # Dirichlet concentration parameters
        S = torch.sum(alpha, dim=-1, keepdim=True)       # Dirichlet strength
        ori_u = self.args.n_actions / S                   # Pre-communication uncertainty u_j

        # --- Step 4: DST Evidence Combination ---
        # Paper Section 4.3, Eq. 5-7: Combine evidence from selected senders
        received_evidence = self.combine_message(evidence, send_target)
        evidence = evidence + received_evidence           # Paper Eq. 9: e_final = e_i + e_hat_i

        # --- Step 5: Post-communication uncertainty ---
        alpha = evidence + 1
        S = torch.sum(alpha, dim=-1, keepdim=True)
        cmb_u = self.args.n_actions / S                   # Post-communication uncertainty u_hat_j

        # Return evidence (used as Q-values) and uncertainty change.
        # Paper Eq. 3: v(m_ij) = u_j - u_hat_j. Here we return (cmb_u - ori_u).
        # Detached because uncertainty change is used as a pseudo-label signal,
        # not for gradient computation.
        return evidence, (cmb_u - ori_u).detach()

    @torch.no_grad()
    def combine_message(self, evidence, send_target):
        """
        Paper Section 4.3: Evidence-Driven Integration using Dempster-Shafer Theory.

        This method:
        1. Converts each agent's evidence to belief masses and uncertainty (Eq. 2)
        2. For each receiving agent, iteratively combines beliefs from all
           selected senders using Dempster's Rule of Combination (Eq. 6)
        3. Converts the combined belief back to evidence (Eq. 8)
        4. Scales and adds to original evidence (Eq. 9)

        The @torch.no_grad() decorator is critical: DST combination is a
        mathematically principled, parameter-free operation. Gradients do NOT
        flow through it. The RL loss (TD error) backpropagates only through
        the evidence encoder, not through the combination.

        Args:
            evidence:    [batch_size, n_agents, n_actions] - each agent's evidence
            send_target: [batch_size, n_agents, n_agents] - binary mask (who sends to whom)

        Returns:
            combined_evidence: [batch_size, n_agents, n_actions] - evidence after DST combination
        """
        # FIX: Use device of evidence tensor instead of hardcoded .cuda()
        device = evidence.device

        batch_evidence = evidence.detach()
        # Transpose: original is [sender, receiver], we need [receiver, sender]
        batch_send_target = send_target.transpose(-2, -1)

        # --- Convert evidence to belief/uncertainty (Paper Eq. 2) ---
        alpha = batch_evidence + 1                                          # alpha_k = e_k + 1
        S = torch.sum(alpha, dim=-1, keepdim=True)                          # S = sum(alpha_k)
        batch_belief = batch_evidence / (S.expand(batch_evidence.shape))    # b_k = e_k / S
        batch_uncertainty = self.args.n_actions / S                          # u = K / S

        # Initialize with vacuous belief (maximum uncertainty = "I know nothing")
        # We flatten the Batch (B) and Receiver (N) dimensions to compute everything simultaneously!
        B = evidence.shape[0]
        N = self.args.n_agents
        A = self.args.n_actions
        
        r_b = torch.zeros((B * N, A), device=device)
        r_u = torch.ones((B * N, 1), device=device)

        # Loop over Senders (only 10 iterations instead of 3200!)
        for sender_idx in range(N):
            # target: did 'sender_idx' send a message to each receiver? Shape: [B*N, 1]
            target = batch_send_target[:, :, sender_idx].reshape(B * N, 1)
            
            # Extract this sender's belief/uncertainty and duplicate it for all receivers
            sender_b = batch_belief[:, sender_idx, :].unsqueeze(1).expand(B, N, A).reshape(B * N, A)
            sender_u = batch_uncertainty[:, sender_idx, :].unsqueeze(1).expand(B, N, 1).reshape(B * N, 1)
            
            # Combine the current receiver beliefs with this sender's belief
            new_b, new_u = self.combine(r_b, r_u, sender_b, sender_u)
            
            # Only apply the new belief if the target actually chose to send (target == 1)
            r_b = torch.where(target == 1, new_b, r_b)
            r_u = torch.where(target == 1, new_u, r_u)

        # Reshape back to [batch_size, n_agents, ...]
        combined_belief = r_b.reshape(B, N, A)
        combined_uncertainty = r_u.reshape(B, N, 1)

        # Convert combined belief back to evidence (Paper Eq. 8)
        combined_evidence = self.belief_to_evidence(combined_belief, combined_uncertainty)

        # Scale received evidence by 0.1 before adding to prevent overwhelming
        # the agent's own evidence with received information.
        combined_evidence = evidence + 0.1 * combined_evidence
        return combined_evidence

    def belief_to_evidence(self, belief, uncertainty):
        """
        Paper Eq. 8: Convert combined belief/uncertainty back to evidence.

        Given combined belief b_k and uncertainty u:
            S_hat = K / u_hat        (recover Dirichlet strength)
            e_hat_k = b_k * S_hat    (recover evidence)

        Args:
            belief:      [*, n_actions] - belief masses
            uncertainty: [*, 1] - overall uncertainty

        Returns:
            e_a: [*, n_actions] - recovered evidence values
        """
        S_a = self.args.n_actions / uncertainty   # Recover Dirichlet strength
        e_a = torch.mul(belief, S_a.expand(belief.shape))  # Recover evidence
        return e_a

    def combine(self, b0, u0, b1, u1):
        """
        Paper Eq. 6: Dempster's Rule of Combination.

        Combines two independent evidence sources (beliefs + uncertainties)
        into a single fused opinion. This is the mathematical heart of T2MAC.

        The combination has three intuitive parts:
        - b0_k * b1_k: Both sources AGREE on action k
        - b0_k * u1:   Source 0 believes k, source 1 is uncertain (defers to source 0)
        - b1_k * u0:   Source 1 believes k, source 0 is uncertain (defers to source 1)

        The conflict C measures how much the sources DISAGREE:
        - C = sum of b0_k * b1_k' for all k != k'

        Properties:
        - Associative and commutative (order doesn't matter)
        - Combined uncertainty always decreases: u_combined <= min(u0, u1)
        - If both uncertain -> cautious output
        - If both agree -> strong output
        - If one uncertain -> follows the confident one

        Args:
            b0, b1: [batch, n_actions] - belief masses from two sources
            u0, u1: [batch, 1] - uncertainties from two sources

        Returns:
            b_a: [batch, n_actions] - combined belief
            u_a: [batch, 1] - combined uncertainty
        """
        # Outer product of beliefs: b0_k * b1_k' for all k, k'
        bb = torch.bmm(b0.view(-1, self.args.n_actions, 1), b1.view(-1, 1, self.args.n_actions))

        # Cross terms: one source is confident, the other defers
        uv1_expand = u1.expand(b0.shape)
        bu = torch.mul(b0, uv1_expand)    # b0_k * u1 (source 1 defers to source 0)
        uv_expand = u0.expand(b0.shape)
        ub = torch.mul(b1, uv_expand)     # b1_k * u0 (source 0 defers to source 1)

        # Calculate conflict C (Paper Eq. 6)
        # C = sum of all off-diagonal elements in the outer product matrix
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)        # Total sum (all k, k')
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)  # Diagonal sum (k == k')
        C = bb_sum - bb_diag                                 # Off-diagonal sum = conflict

        # Combined belief (Paper Eq. 6)
        # b_a_k = (b0_k * b1_k + b0_k * u1 + b1_k * u0) / (1 - C)
        b_a = (torch.mul(b0, b1) + bu + ub) / ((1 - C).view(-1, 1).expand(b0.shape))

        # Combined uncertainty (Paper Eq. 6)
        # u_a = (u0 * u1) / (1 - C)
        u_a = torch.mul(u0, u1) / ((1 - C).view(-1, 1).expand(u0.shape))

        return b_a, u_a

    def init_hidden(self):
        """
        Initialize hidden state to zeros.
        Uses the marine adapter weight tensor to automatically inherit the correct device.
        """
        return self.adapters["marine"].weight.new_zeros(1, self.args.rnn_hidden_dim)
