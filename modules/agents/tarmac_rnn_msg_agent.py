import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TarMACRnnMsgAgent(nn.Module):
    """RNN agent with TarMAC signature-query-value communication."""

    def __init__(self, input_dim, args):
        super().__init__()

        if args.n_key != args.n_query:
            raise ValueError("TarMAC requires n_key == n_query for dot-product attention.")

        self.args = args
        self.n_rounds = max(1, getattr(args, "tarmac_rounds", 1))
        self.include_self = getattr(args, "tarmac_self_attention", True)

        self.fc1 = nn.Linear(input_dim + args.n_value, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.fc_key = nn.Linear(args.rnn_hidden_dim, args.n_key)
        self.fc_query = nn.Linear(args.rnn_hidden_dim, args.n_query)
        self.fc_value = nn.Linear(args.rnn_hidden_dim, args.n_value)
        self.fc_comm_update = nn.Linear(args.rnn_hidden_dim + args.n_value, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.last_attn = None

    def forward(self, x, hidden, comm=None):
        if comm is None:
            batch_agents = x.shape[0]
            comm = x.new_zeros(batch_agents, self.args.n_value)
        else:
            comm = comm.reshape(-1, self.args.n_value)

        x = torch.cat([x, comm], dim=-1)
        x = F.relu(self.fc1(x))
        h_in = hidden.reshape(-1, self.args.rnn_hidden_dim)
        h_out = self.rnn(x, h_in)
        return h_out.view(-1, self.args.n_agents, self.args.rnn_hidden_dim)

    def q_without_communication(self, hidden):
        return self.fc2(hidden)

    def communicate(self, hidden):
        key = self.fc_key(hidden)
        value = self.fc_value(hidden)
        query = self.fc_query(hidden)
        return key, value, query

    def aggregate(self, query, key, value, hidden):
        del query, key, value

        round_hidden = hidden
        attentions = []

        message = None
        for _ in range(self.n_rounds):
            key, value, query = self.communicate(round_hidden)
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.args.n_key)

            if not self.include_self:
                eye = torch.eye(self.args.n_agents, device=scores.device, dtype=torch.bool)
                scores = scores.masked_fill(eye.unsqueeze(0), -1e9)

            attn_weights = F.softmax(scores, dim=-1)
            message = torch.bmm(attn_weights, value)
            round_hidden = torch.tanh(self.fc_comm_update(torch.cat([round_hidden, message], dim=-1)))
            attentions.append(attn_weights)

        self.last_attn = torch.stack(attentions, dim=1)
        return self.fc2(round_hidden), message

    def init_hidden(self):
        return self.fc1.weight.new_zeros(1, self.args.rnn_hidden_dim)
