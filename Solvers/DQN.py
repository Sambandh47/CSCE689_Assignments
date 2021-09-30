import os
import random
from collections import deque
import tensorflow as tf
from keras import backend as bk
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import huber_loss
import numpy as np
from scipy.special import softmax #importing softmax function from scipy used in the policy function

from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


class DQN(AbstractSolver):
    def __init__(self,env,options):
        assert (str(env.action_space).startswith('Discrete') or
        str(env.action_space).startswith('Tuple(Discrete')), str(self) + " cannot handle non-discrete action spaces"
        super().__init__(env,options)
        self.model = self._build_model()
        self.target_model = self._build_model()
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        # Add required fields          #
        ################################
        self.replay_memory=[] #initializing an array of replay memory
        self.C = self.options.update_target_estimator_every #updating target estimator in every step



    def _build_model(self):
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        layers = self.options.layers
        # Neural Net for Deep-Q learning Model
        model = Sequential()

        model.add(Dense(layers[0], input_dim=state_size, activation='relu'))
        for l in layers:
            model.add(Dense(l, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss=huber_loss,
                      optimizer=Adam(lr=self.options.alpha))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def make_epsilon_greedy_policy(self):
        """
        Creates an epsilon-greedy policy based on the given Q-function approximator and epsilon.

        Returns:
            A function that takes a state as input and returns a vector
            of action probabilities.
        """
        nA = self.env.action_space.n

        def policy_fn(state):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            random_number = np.random.random_sample() #generating random numbers
            if random_number>=self.options.epsilon: #checking the value of random no generated if it is greater than epsilon
                next_action = self.model.predict(np.array([state,]))[0] #selecting the next action
                return np.exp(next_action)/np.exp(next_action).sum(axis=0) #returning the probability of selecting the action

            A = np.random.rand(nA) #selecting a random action
            return np.exp(A)/np.exp(A).sum(axis=0) #returning the probability of the action
        return policy_fn

    def train_episode(self):
        """
        Perform a single episode of the Q-Learning algorithm for off-policy TD
        control using a DNN Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Use:
            self.options.experiment_dir: Directory to save DNN summaries in (optional)
            self.options.replay_memory_size: Size of the replay memory
            self.options.update_target_estimator_every: Copy parameters from the Q estimator to the
                target estimator every N steps
            self.options.batch_size: Size of batches to sample from the replay memory
            self.env: OpenAI environment.
            self.options.gamma: Gamma discount factor.
            self.options.epsilon: Chance the sample a random action. Float betwen 0 and 1.
            new_state, reward, done, _ = self.step(action): To advance one step in the environment
            state_size = self.env.observation_space.shape[0]
            self.model: Q network
            self.target_model: target Q network
            self.update_target_model(): update target network weights = Q network weights
        """

        # Reset the environment
        state = self.env.reset()

        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        next_state = state #defining the next state
        self.policy = self.make_epsilon_greedy_policy() #defining the policy
        while True:
            state = next_state #storing the next state as current state
            action_probabilities = self.policy(state) #generating the action probabilities for the state
            action = np.argmax(action_probabilities) #selecting the best action
            next_state, reward, done, _ = self.step(action)
            replay_tuple = (state, action, reward, done, next_state) #storing the replay tuple
            if len(self.replay_memory)==self.options.replay_memory_size: #popping the element if there is just one element in tuple
                self.replay_memory.pop(0)
            self.replay_memory.append(replay_tuple) #appending the generated tuple in the replay memory
            mini_batch=random.choices(self.replay_memory,k=self.options.batch_size) #generating a mini batch randomly from the set of replay tuples
            train_x_states = []
            train_y_rewards = []
            for batch in mini_batch:
                train_x_states.append(batch[0]) #updating the train x states
                train_y_reward=self.model.predict(np.array([batch[0],]))[0] #predicting the reward for the 1st batch
                if (batch[3] == True):
                    train_y_reward[batch[1]] = batch[2]
                else:
                    train_y_reward[batch[1]] = batch[2] + self.options.gamma * max(self.target_model.predict(np.array([batch[4],]))[0])
                train_y_rewards.append(train_y_reward)

            self.model.train_on_batch(np.array(train_x_states),np.array(train_y_rewards)) #passing the x states and y rewards to the NN model
            self.C = self.C-1 #decrementing the value of C
            if self.C==0:
                self.update_target_model()
                self.C=self.options.update_target_estimator_every

            if done:
                break
                

    def __str__(self):
        return "DQN"

    def plot(self,stats):
        plotting.plot_episode_stats(stats)

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values.


        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """
        nA = self.env.action_space.n

        def policy_fn(state):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            #action_probabilities = self.target_model.predict(np.array([state,])) #generating the action probabilities
            #return softmax(action_probabilities[0]) #running a softmax function on the vector
            model_action_probabilities = self.model.predict(np.array([state, ]))
            max_action_q_value = float("-inf")
            action_probabilities = np.zeros(nA)
            best_action = 0
            for action in range(nA):
                if max_action_q_value < model_action_probabilities[0][action]:
                    max_action_q_value = model_action_probabilities[0][action]
                    best_action = action

            action_probabilities[best_action] = 1
            return action_probabilities

        return policy_fn