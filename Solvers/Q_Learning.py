from collections import defaultdict

import numpy as np
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


class QLearning(AbstractSolver):

    def __init__(self, env, options):
        assert (str(env.action_space).startswith('Discrete') or
                str(env.action_space).startswith('Tuple(Discrete')), str(
            self) + " cannot handle non-discrete action spaces"
        super().__init__(env, options)
        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        # The policy we're following
        self.policy = self.make_epsilon_greedy_policy()

    def train_episode(self):
        """
        Run a single episode of the Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
        while following an epsilon-greedy policy

        Use:
            self.env: OpenAI environment.
            self.options.gamma: Gamma discount factor.
            self.options.alpha: TD learning rate.
            self.options.epsilon: Chance to sample a random action. Float between 0 and 1.
            new_state, reward, done, _ = self.step(action): To advance one step in the environment
        """

        # Reset the environment
        state = self.env.reset()


        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        while True:
            action_probabilities  = self.policy(state) #generating the action probabilities for each state
            action = np.random.choice(np.arange(len(action_probabilities)),p=action_probabilities) #choosing the action
            next_state, reward, done, _ = self.step(action) #generating the next state and reward due to a particular action
            best_next_action = np.argmax(self.Q[next_state]) #choosing the best action based on the next state
            td_target = reward + self.options.gamma * self.Q[next_state][best_next_action] #generating the target value taking into account the next state and the best next action
            td_delta = td_target-self.Q[state][action] #generating the difference
            self.Q[state][action] = self.Q[state][action]+self.options.alpha*td_delta #updating the Q values

            if done:
                break

            state = next_state #updating the state


    def __str__(self):
        return "Q-Learning"

    def plot(self, stats):
        plotting.plot_episode_stats(stats)

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
            no_of_actions = self.env.action_space.n
            action_probabilities = np.zeros(no_of_actions)
            best_action = 0
            max_action_q_value = float("-inf")
            for a in range(no_of_actions):
                if max_action_q_value<self.Q[state][a]:
                    max_action_q_value=self.Q[state][a]
                    best_action=a
            action_probabilities[best_action]=1
            return action_probabilities

        return policy_fn

    def make_epsilon_greedy_policy(self):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.

        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.

        """
        nA = self.env.action_space.n

        def policy_fn(observation):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            action = (np.ones(nA, dtype=float) * self.options.epsilon)/nA
            best_action = np.argmax(self.Q[observation])
            action[best_action] = action[best_action]+(1-self.options.epsilon)
            return  action

        return policy_fn


class ApproxQLearning(QLearning):
    # MountainCar-v0
    def __init__(self, env, options):
        self.estimator = Estimator(env)
        super().__init__(env, options)

    def train_episode(self):
        """
        Run a single episode of the approximated Q-Learning algorithm: Off-policy TD control.
        Finds the optimal greedy policy while following an epsilon-greedy policy

        Use:
            self.env: OpenAI environment.
            self.options.gamma: Gamma discount factor.
            self.options.alpha: TD learning rate.
            self.options.epsilon: Chance to sample a random action. Float between 0 and 1.
            self.estimator: The Q-function approximator
            new_state, reward, done, _ = self.step(action): To advance one step in the environment
        """

        # Reset the environment
        state = self.env.reset()

        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        while True:
            action_probabilities = self.policy(state) #generating the action probabilities for each state
            action = np.random.choice(np.arange(len(action_probabilities)),p=action_probabilities) #choosing the action
            next_state, reward, done, _ = self.step(action) #generating the next state and reward due to a particular action
            next_q_values = self.estimator.predict(next_state) #generating the array of next set of q values depending on the next state
            td_target = reward + self.options.gamma * np.max(next_q_values) #generating the target value taking into account the max q values generated from next state and the value of gamma
            self.estimator.update(state,action,td_target)

            if done:
                break

            state = next_state #updating the state

    def __str__(self):
        return "Approx Q-Learning"

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values from self.estimator

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """
        nA = self.env.action_space.n

        def policy_fn(observation):

            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            action_probabilities = np.zeros(nA) #initializing the action probabilities
            best_action = 0 #initializing the best action to zero
            max_action_q_value = float("-inf") #initialzing the q value of the best action
            for a in range(nA): #iterating over the action space
                if max_action_q_value < self.estimator.predict(observation)[a]:
                    max_action_q_value = self.estimator.predict(observation)[a] #updating the q value for the best action
                    best_action = a #updating the best action
            action_probabilities[best_action] = 1 #initializing the action probabilities of the best action taken to 1
            return action_probabilities


        return policy_fn

    def make_epsilon_greedy_policy(self):
        """
        Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

        Use:
            self.options.epsilon: The probability to select a random action . float between 0 and 1.
            self.env.action_space.n: Number of actions in the environment.
            self.estimator: The Q-values estimator

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
            if random_number < self.options.epsilon:  #choosing a random probability and comparing it with the value of epsilon
                return np.ones(nA) / nA

            greedy_policy = self.create_greedy_policy() #updating the greedy policy
            return greedy_policy(observation)

        return policy_fn

    def plot(self, stats):
        plotting.plot_cost_to_go_mountain_car(self.env, self.estimator)
        plotting.plot_episode_stats(stats, smoothing_window=25)


class Estimator:
    """
    Value Function approximator. Don't change!
    """

    def __init__(self, env):
        # Feature Preprocessing: Normalize to zero mean and unit variance
        # We use a few samples from the observation space to do this
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # Used to convert a state to a featurizes represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        self.featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
        self.featurizer.fit(self.scaler.transform(observation_examples))
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)

    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]

    def predict(self, s, a=None):
        """
        Makes value function predictions.

        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for

        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.

        """
        features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]

    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])
