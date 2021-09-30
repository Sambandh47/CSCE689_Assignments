from collections import defaultdict

import numpy as np

from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


class Sarsa(AbstractSolver):

    def __init__(self,env,options):
        assert str(env.observation_space).startswith('Discrete'), str(self) + \
                                                                  " cannot handle non-discrete state spaces"
        assert (str(env.action_space).startswith('Discrete') or
        str(env.action_space).startswith('Tuple(Discrete')), str(self) + " cannot handle non-discrete action spaces"
        super().__init__(env,options)
        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        # The policy we're following
        self.policy = self.make_epsilon_greedy_policy()

    def train_episode(self):
        """
        Run one episode of the SARSA algorithm: On-policy TD control.

        Use:
            self.env: OpenAI environment.
            self.options.gamma: Gamma discount factor.
            self.options.alpha: TD learning rate.
            self.options.epsilon: Chance the sample a random action. Float betwen 0 and 1.

        """

        # Reset the environment
        state = self.env.reset()

        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        action_probabilities = self.policy(state) #generating all the action probabilities for a particular state by running the policy function over it
        action = np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities) #choosing a random action based on action probability

        while True:
            next_state, reward, done, _ = self.step(action) #geneating the next state and reward by choosing the action on a state
            next_action_probabilities = self.policy(next_state) #generating the next set of action probabilities by running the policy function over it
            next_action = np.random.choice(np.arange(len(next_action_probabilities)), p=next_action_probabilities) #choosing the next action based on next set of action state probabilities
            td_target = reward + self.options.gamma*self.Q[next_state][next_action] #generating the target
            td_delta = td_target-self.Q[state][action] #generating the temporal difference
            self.Q[state][action] = self.Q[state][action] + (self.options.alpha*td_delta) #generating the Q value for a state-action pair
            action = next_action #updating the action
            state = next_state #updating the state
            if done:
                break


    def __str__(self):
        return "Sarsa"

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
            action_probabilities = np.zeros(nA) #initializing all the action probabilities
            action = 0 #initializng the best action to be zero
            max_action_q_value = float("-inf") #initializing the q value corresponding to action with max probability
            for a in range(nA): #looping through all the actions
                if max_action_q_value < self.Q[state][a]: #if statement to update the q value corresponding to the max action
                    max_action_q_value = self.Q[state][a]
                    action = a #updating the action
            action_probabilities[action] = 1 #initializing the action probabilities for the chosen action to 1
            return action_probabilities

        return policy_fn

    def make_epsilon_greedy_policy(self):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.

        Use:
            self.Q: A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA
            self.options.epsilon: The probability to select a random action . float between 0 and 1.
            self.env.action_space.n: Number of actions in the environment.

        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.

        """
        nA = self.env.action_space.n

        def policy_fn(observation):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            random_number = np.random.random_sample()
            if random_number < self.options.epsilon: #choosing a random probability and comparing it with the value of epsilon
                return np.ones(nA)/nA

            greedy_policy = self.create_greedy_policy()
            return greedy_policy(observation)

        return policy_fn

    def plot(self,stats):
        plotting.plot_episode_stats(stats)
