import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 1e-2     # L2 weight decay
EPS = 1.0               # Noise Decay - not used
EPS_DECAY = 1e-6

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, num_agents, state_size, action_size, random_seed, lr_a=LR_ACTOR, lr_c=LR_CRITIC, weight_decay=WEIGHT_DECAY, fc1_units=400, fc2_units=300):
        """Initialize an Agent object.

        Params
        ======
            num_agents (int): number of agents
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            lr_a (float): learning rate for actor
            lr_c (float): learning rate for critic
            weight_decay: L2 weight decay
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """

        print('Agent Parameters:')
        print('Number of agents:', num_agents)
        print('lr_a:', lr_a)
        print('lr_c:', lr_c)
        print('weight decay:', weight_decay)
        print('fc1_units:', fc1_units)
        print('fc2_units:', fc2_units)

        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        #self.epsilon = EPS

        # Actor Network (w/ Target Network)
        self.actor_local = [Actor(
            state_size, action_size, random_seed, fc1_units=fc1_units, fc2_units=fc2_units).to(device) for _ in range(num_agents)]

        self.actor_target = [Actor(
            state_size, action_size, random_seed, fc1_units=fc1_units, fc2_units=fc2_units).to(device) for _ in range(num_agents)]

        # Critic Network (w/ Target Network)

        self.critic_local = [Critic(
            state_size * num_agents, action_size * num_agents, random_seed, fcs1_units=fc1_units, fc2_units=fc2_units).to(device) for _ in range(num_agents)]

        self.critic_target = [Critic(
            state_size * num_agents, action_size * num_agents, random_seed, fcs1_units=fc1_units, fc2_units=fc2_units).to(device) for _ in range(num_agents)]

        self.actor_optimizer = [optim.Adam(
            self.actor_local[agent_num].parameters(), lr=lr_a) for agent_num in range(num_agents)]

        self.critic_optimizer = [optim.Adam(
            self.critic_local[agent_num].parameters(), lr=lr_c, weight_decay=weight_decay) for agent_num in range(num_agents)]

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(
            num_agents, state_size, action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and timestep % self.num_agents == 0:
            for i in range(self.num_agents):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""

        action = np.zeros([2, 2])
        for i in range(self.num_agents):
            new_state = state[i]
            new_state.resize(1, 24)  # reshape for batch processing
            new_state = torch.from_numpy(new_state).float().to(device)
            self.actor_local[i].eval()
            with torch.no_grad():
                action[i] = self.actor_local[i](new_state).cpu().data.numpy()
            self.actor_local[i].train()
            if add_noise:
                action[i] += self.noise.sample()
                #action += self.epsilon*self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, full_actions, full_states, next_full_states = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = [self.actor_target[agent_num](
            next_states[agent_num]) for agent_num in range(self.num_agents)]

        actions_next_combined = torch.cat(actions_next, dim=1)

        with torch.no_grad():
            Q_targets_next = [self.critic_target[agent_num](
                next_full_states, actions_next_combined) for agent_num in range(self.num_agents)]
        # Compute Q targets for current states (y_i)
        Q_targets = [rewards[agent_num] + (gamma * Q_targets_next[agent_num] * (
            1 - dones[agent_num])) for agent_num in range(self.num_agents)]
        # Compute critic loss
        Q_expected = [self.critic_local[agent_num](
            full_states, full_actions) for agent_num in range(self.num_agents)]
        critic_loss = [F.mse_loss(Q_expected[agent_num], Q_targets[agent_num])
                       for agent_num in range(self.num_agents)]

        # Minimize the loss
        for agent_num in range(self.num_agents):
            self.critic_optimizer[agent_num].zero_grad()
            if agent_num == 0:
                critic_loss[agent_num].backward(retain_graph=True)
            else:
                critic_loss[agent_num].backward()
            # torch.nn.utils.clip_grad_norm_(
            #     self.critic_local[agent_num].parameters(), 1)  # can possibly help improve performance
            self.critic_optimizer[agent_num].step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = [self.actor_local[agent_num](
            states[agent_num]) for agent_num in range(self.num_agents)]

        actions_pred_combined = torch.cat(actions_pred, dim=1)

        actor_loss = [-self.critic_local[agent_num]
                      (full_states, actions_pred_combined).mean() for agent_num in range(self.num_agents)]

        # Minimize the loss
        for agent_num in range(self.num_agents):
            self.actor_optimizer[agent_num].zero_grad()
            if agent_num == 0:
                actor_loss[agent_num].backward(retain_graph=True)
            else:
                actor_loss[agent_num].backward()
            self.actor_optimizer[agent_num].step()

        # ----------------------- update target networks ----------------------- #
        for agent_num in range(self.num_agents):
            self.soft_update(
                self.critic_local[agent_num], self.critic_target[agent_num], TAU)
            self.soft_update(
                self.actor_local[agent_num], self.actor_target[agent_num], TAU)

        # added
        #self.epsilon -= EPS_DECAY
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, num_agents, state_size, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            num_agents (int): number of agents presetn
            state_size (int): state size
            action_size (int): action size
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """

        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = [torch.from_numpy(
            np.vstack([e.state[agent_num] for e in experiences if e is not None])).float().to(device) for agent_num in range(self.num_agents)]

        actions = [torch.from_numpy(
            np.vstack([e.action[agent_num] for e in experiences if e is not None])).float().to(device) for agent_num in range(self.num_agents)]

        rewards = [torch.from_numpy(
            np.vstack([e.reward[agent_num] for e in experiences if e is not None])).float().to(device) for agent_num in range(self.num_agents)]

        next_states = [torch.from_numpy(np.vstack(
            [e.next_state[agent_num] for e in experiences if e is not None])).float().to(device) for agent_num in range(self.num_agents)]

        dones = [torch.from_numpy(np.vstack(
            [e.done[agent_num] for e in experiences if e is not None]).astype(np.uint8)).float().to(device) for agent_num in range(self.num_agents)]

        full_actions = torch.from_numpy(
            np.vstack([e.action.reshape(1, self.num_agents*self.action_size) for e in experiences if e is not None])).float().to(device)
        full_states = torch.from_numpy(
            np.vstack([e.state.reshape(1, self.num_agents*self.state_size) for e in experiences if e is not None])).float().to(device)
        next_full_states = torch.from_numpy(np.vstack(
            [e.next_state.reshape(1, self.num_agents*self.state_size) for e in experiences if e is not None])).float().to(device)

        return (states, actions, rewards, next_states, dones, full_actions, full_states, next_full_states)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
