# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

isaac_gym_joint_names = [
    'Left_Hip_Pitch',
    'Left_Hip_Roll',
    'Left_Hip_Yaw',
    'Left_Knee_Pitch',
    'Left_Ankle_Pitch',
    'Left_Ankle_Roll',
    'Right_Hip_Pitch',
    'Right_Hip_Roll',
    'Right_Hip_Yaw',
    'Right_Knee_Pitch',
    'Right_Ankle_Pitch',
    'Right_Ankle_Roll'
]

isaac_lab_joint_names = [
    'Left_Hip_Pitch',
    'Right_Hip_Pitch',
    'Left_Hip_Roll',
    'Right_Hip_Roll',
    'Left_Hip_Yaw',
    'Right_Hip_Yaw',
    'Left_Knee_Pitch',
    'Right_Knee_Pitch',
    'Left_Ankle_Pitch',
    'Right_Ankle_Pitch',
    'Left_Ankle_Roll',
    'Right_Ankle_Roll'
]

gym2lab_mapping = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]
lab2gym_mapping = [gym2lab_mapping.index(i) for i in range(12)]

# History Encoder
class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 3 * channel_size), self.activation_fn,
        )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
            nn.Linear(channel_size * 3, output_size), self.activation_fn
        )

    def forward(self, obs):
        # (batch_size, T, num_prop)
        nd = obs.shape[0]
        T = self.tsteps
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, mlp_input_dim_a, actor_hidden_dims, activation, num_actions, num_priv, num_hist, num_prop, priv_encoder_dims):
        super().__init__()
        self.num_actions = num_actions
        self.num_priv = num_priv
        self.num_hist = num_hist
        self.num_prop = num_prop

        # Convert gym2lab_mapping and lab2gym_mapping to tensors
        self.register_buffer("gym2lab_mapping", torch.tensor([0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11], dtype=torch.long))
        self.register_buffer("lab2gym_mapping", torch.tensor([self.gym2lab_mapping.tolist().index(i) for i in range(12)], dtype=torch.long))

        # Dummy History Encoder
        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, priv_encoder_dims[-1])
        
        # Actor model (policy network)
        actor_layers = [nn.Linear(num_prop + priv_encoder_dims[-1], actor_hidden_dims[0]), activation]
        for i in range(len(actor_hidden_dims) - 1):
            actor_layers.extend([nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]), activation])
        actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        self.actor = nn.Sequential(*actor_layers)

    def process_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                obs[..., :12],
                obs[..., 12:24].index_select(-1, self.gym2lab_mapping),
                obs[..., 24:36].index_select(-1, self.gym2lab_mapping),
                obs[..., 36:48].index_select(-1, self.gym2lab_mapping)
            ),
            dim=-1,
        )

    def infer_hist_latent(self, obs):
        return self.history_encoder(obs.view(-1, self.num_hist, self.num_prop))

    @torch.jit.export
    def act_mj(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.process_obs(obs)
        latent = self.infer_hist_latent(obs)
        actor_input = torch.cat([obs[:, -1], latent], dim=1)
        return self.actor(actor_input).index_select(-1, self.lab2gym_mapping)

if __name__ == "__main__":
    # Script the act_mj function
    actor_model = Actor(
        mlp_input_dim_a=36,
        actor_hidden_dims=[256, 256],
        activation=nn.ReLU(),
        num_actions=12,
        num_priv=10,
        num_hist=20,
        num_prop=48,
        priv_encoder_dims=[64, 20]
    )

    scripted_actor = torch.jit.script(actor_model)
    obs = torch.randn(1, 20, 48)  # (batch_size, history_length, num_prop)

    # Call act_mj on the scripted model
    actions = scripted_actor.act_mj(obs)
    # actions = actor_model.act_mj(obs)
