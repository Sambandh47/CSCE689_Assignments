from collections import defaultdict, OrderedDict

import numpy as np

from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


class MonteCarlo(AbstractSolver):

    def __init__(self,env,options):
        assert (str(env.observation_space).startswith('Discrete') or
        str(env.observation_space).startswith('Tuple(Discrete')), str(self) + " cannot handle non-discrete state spaces"
        assert (str(env.action_space).startswith('Discrete') or
        str(env.action_space).startswith('Tuple(Discrete')), str(self) + " cannot handle non-discrete action spaces"
        super().__init__(env,options)

        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.env = env

        ################################
        #   YOUR IMPLEMENTATION HERE   #
        # Add required fields          #
        ################################
        self.returns = defaultdict(list) #initializes the return values in a dictionary

    def train_episode(self):
        """
            Run a single episode for Monte Carlo Control using Epsilon-Greedy policies.

            Use:
                self.options.env: OpenAI gym environment.
                self.options.gamma: Gamma discount factor.
                self.options.epsilon: Chance the sample a random action. Float betwen 0 and 1.
                new_state, reward, done, _ = self.step(action): To advance one step in the environment
        """

        # Reset the environment
        state = self.env.reset()

        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        policy_fn = self.make_epsilon_greedy_policy()
        episode = [] #Declaring episode as an empty list
        while True:
            action_values = policy_fn(state)  #generating the action values
            best_action = np.random.choice(np.arange(len(action_values)), p=action_values) #selecting the best action value from the list of action values
            new_state, reward, done, _ = self.step(best_action) #generating the next state and reward from the best action on a state
            episode.append((state, best_action, reward))  #updating the episode
            state = new_state
            if done:
                break

        G = 0
        for i in reversed(range(len(episode))): #traversing through all episodes starting from last episode
            state, action, reward = episode[i] #extracting the state,action and reward for a particular episode
            state_action_pair = (state, action) #generating a state-action pair
            G = (G * self.options.gamma) + reward #updating the G value
            if (state_action_pair in [(x[0], x[1]) for x in episode[:i]]) == False:
                self.returns[(state, action)].append(G) #appending G
                self.Q[state][action] = np.mean(self.returns[(state, action)]) #average
                policy = self.make_epsilon_greedy_policy() #updating the policy


    def __str__(self):
        return "Monte Carlo"

    def make_epsilon_greedy_policy(self):
        """
        Creates an epsilon-greedy policy based on a given Q-estimates and epsilon.

        Use:
            self.Q: A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA
            self.options.epsilon: Chance the sample a random action. Float betwen 0 and 1.
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
            action_values = []
            best_action_value = max(self.Q[observation]) #generating the best action value by taking the max of Q values for each observation
            for i in self.Q[observation]:
                if(i==best_action_value): #updating the weights to best action value
                    action_values.append(1-self.options.epsilon+(self.options.epsilon/nA))
                else:
                    action_values.append(self.options.epsilon/nA)
            return np.exp(action_values)/np.exp(action_values).sum(axis=0)


        return policy_fn

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values.

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """

        def policy_fn(state):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            action = np.zeros_like(self.Q[state], dtype=float)
            best_action = np.argmax(self.Q[state])
            action[best_action] = 1.0
            return action

        return policy_fn

    def plot(self,stats):
        # For plotting: Create value function from action-value function
        # by picking the best action at each state
        V = defaultdict(float)
        for state, actions in self.Q.items():
            action_value = np.max(actions)
            V[state] = action_value
        plotting.plot_value_function(V, title="Final Value Function")


class OffPolicyMC(MonteCarlo):
    def __init__(self,env,options):
        assert (str(env.action_space).startswith('Discrete') or
        str(env.action_space).startswith('Tuple(Discrete')), str(self) + " cannot handle non-discrete action spaces"
        super().__init__(env,options)

        # The cumulative denominator of the weighted importance sampling formula
        # (across all episodes)
        self.C = defaultdict(lambda: np.zeros(env.action_space.n))

        # Our greedily policy we want to learn about
        self.target_policy = self.create_greedy_policy()
        # Our behavior policy we want to learn from
        self.behavior_policy = self.create_random_policy()

    def train_episode(self):
        """
            Run a single episode of Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.

            Use:
                self.options.env: OpenAI gym environment.
                self.options.gamma: Gamma discount factor.
                new_state, reward, done, _ = self.step(action): To advance one step in the environment
        """

        # Reset the environment
        state = self.env.reset()

        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        episode = []
        while True:
            action_values = self.behavior_policy(state)
            best_action = np.random.choice(np.arange(len(action_values)), p=action_values)
            new_state, reward, done, _ = self.step(best_action)
            episode.append((state, best_action, reward))
            if done:
                break
            state = new_state

        G = 0
        W = 1
        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            G = self.options.gamma * G + reward
            self.C[state][action] = W + self.C[state][action]
            self.Q[state][action] = self.Q[state][action] + ((W/self.C[state][action]) * (G - self.Q[state][action])) #updating Q values for each state-action pair
            if action != np.argmax(self.target_policy(state)):
                break
            W = W * 1./self.behavior_policy(state)[action]


        
    def create_random_policy(self):
        """
        Creates a random policy function.

        Use:
            self.env.action_space.n: Number of actions in the environment.

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities
        """
        nA = self.env.action_space.n
        A = np.ones(nA, dtype=float) / nA

        def policy_fn(observation):
            return A

        return policy_fn

    def __str__(self):
        return "MC+IS"
