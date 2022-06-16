from mimetypes import init
import os
from pickletools import optimize
from re import A
import re
from turtle import forward
import torch as T
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt =1e-2, x0 = None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)

        self.x_prev = x
        return x
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros(self.mem_size, n_actions)
        self.reward_memory = np.zeros(self.mem_size)
        self.term_memory = np.zeros(self.mem_size, dtype=np.int)

    def store_transition(self, state, action, reward, _state, done):
        idx = self.mem_ctr % self.mem_size
        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.new_state_memory[idx] = _state
        self.term_memory[idx] = done # 0 if done, 1 if not -> the equation of the updating of the weight depends on this value
        self.mem_ctr += 1


    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terms = self.term_memory[batch]

        return states, actions, rewards, new_states, terms


class CriticNetwork(nn.Module):
    def __init__(self,  beta, input_shape, n_actions , name,  chkpt_dir= 'tmp/ddpg', fc1_dims=400, fc2_dims=300):
        super(CriticNetwork, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.fc1_units = fc1_dims
        self.fc2_units = fc2_dims
        self.beta = beta
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.checkpoint_file = os.path.join(self.chkpt_dir, self.name + '_ddpg.pth')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)

        print(*self.input_shape, self.fc1_dims, self.fc1.weight.data.size()[0])
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0]) # equal to 1/sqrt(self.input_shape[0] * self.fc1_dims)
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.bn1 = nn.BatchNorm1d(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims, init)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.BatchNorm1d(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)

        f3 = 0.003

        self.q = nn.Linear(self.fc2_dims, 1)
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)


    def forward(self, state, action):
        x_1 = T.nn.ReLU(self.bn1(self.fc1(state)))
        x_2 = self.bn2(self.fc2(x_1))

        action_value = T.nn.ReLU(self.action_value(action))
        state_action_value = T.nn.ReLU(T.add(x_2, action_value))

        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print('Saving checkpoint...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('Loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_shape, n_actions, name, chkpt_dir= 'tmp/ddpg', fc1_dims=400, fc2_dims=300):
        super(ActorNetwork, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.checkpoint_file = os.path.join(self.chkpt_dir, self.name + '_ddpg_actor.pth')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.bn1 = nn.BatchNorm1d(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims, init)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.BatchNorm1d(self.fc2_dims)


        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)


        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x_1 = T.nn.ReLU(self.bn1(self.fc1(state)))
        x_2 = T.nn.ReLU(self.bn2(self.fc2(x_1)))

        stiring = T.tanh(self.mu(x_2))

        return stiring

    def save_checkpoint(self):
        print('Saving checkpoint...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('Loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))

    
class Agent(object):
    def __init__(self, alpha, beta, input_shape, tau, env, gamma = 0.99, n_actions = 2, max_size = 100000, layer1_size = 400, layer2_size = 300, batch_size = 32):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_shape, n_actions)
        self.batch_size = batch_size

        self.actor = ActorNetwork(alpha, input_shape, fc1_dims = layer1_size,
                                  fc2_dims = layer2_size, n_actions = n_actions, name='Actor')

        self.actor_target = ActorNetwork(alpha, input_shape, fc1_dims = layer1_size,
                                  fc2_dims = layer2_size, n_actions = n_actions, name='Target_actor')

        self.critic = CriticNetwork(beta, input_shape, n_actions, name='Critic',
                                    fc1_dims=layer1_size, fc2_dims=layer2_size)

        self.critic_target = CriticNetwork(beta, input_shape, n_actions, name='Target_critic',
                                    fc1_dims=layer1_size, fc2_dims=layer2_size)


        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_network_parameters(tau)
        # self.fc2 = nn.Linear(self.fc1_units + self.n_actions, self.fc2_units)

    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise()).to(self.actor.device)

        self.actor.train()

        return mu_prime.cpu().detach().numpy() # cannot pass on a tensor to the openAI gym environment

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self):
        if self.memory.mem_ctr > self.batch_size:
            state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

            reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
            done = T.tensor(done, dtype=T.float).to(self.critic.device)
            new_state = T.tensor(next_state, dtype=T.float).to(self.critic.device)
            action = T.tensor(action, dtype=T.float).to(self.critic.device)

            state = T.tensor(state, dtype=T.float).to(self.critic.device)

            self.actor_target.eval()
            self.critic_target.eval()

            target_actions = self.actor_target(new_state)
            critic_value_ = self.critic_target(new_state, target_actions)

            critic_value = self.critic(state, action)

            target = []

            for j in range(self.batch_size):
                target.append(reward[j] + self.gamma * critic_value_[j] * (1 - done[j]))
            
            target = T.tensor(target, dtype=T.float).to(self.critic.device)
            target = target.view(-1, 1)


            self.critic.train()
            self.critic_optim.zero_grad()
            critic_loss = F.mse_loss(critic_value, target)
            critic_loss.backward()
            self.critic.optimizer.step()

            self.critic.eval()
            self.actor.optimizer.zero_grad()
            mu = self.actor(state)
            self.actor.train()
            actor_loss = -self.critic(state, mu)
            actor_loss = T.mean(actor_loss)
            actor_loss.backward()
            self.actor.optimizer.step()

            self.update_network_parameters(tau = 1)

    def update_network_parameters(self, tau = None):
        if tau is None:
            tau = self.tau
        
        # actor_parameters = self.actor.named_parameters()
        # actor_target_parameters = self.actor_target.named_parameters()
        # critic_parameters = self.critic.named_parameters()
        # critic_target_parameters = self.critic_target.named_parameters()

        # actor_state_dict = dict(actor_parameters)
        # actor_target_state_dict = dict(actor_target_parameters)
        # critic_state_dict = dict(critic_parameters)
        # critic_target_state_dict = dict(critic_target_parameters)



        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1 - self.tau) + param.data * self.tau
            )

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                 target_param.data * (1 - self.tau) + param.data * self.tau
            )  

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.actor_target.load_checkpoint()
        self.critic_target.load_checkpoint()










