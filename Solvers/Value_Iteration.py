import numpy as np
import heapq

from Solvers.Abstract_Solver import AbstractSolver, Statistics


class ValueIteration(AbstractSolver):

    def __init__(self,env,options):
        assert str(env.observation_space).startswith( 'Discrete' ), str(self) + \
                                                                    " cannot handle non-discrete state spaces"
        assert str(env.action_space).startswith('Discrete'), str(self) + " cannot handle non-discrete action spaces"
        super().__init__(env,options)
        self.V = np.zeros(env.nS)

    def train_episode(self):
        """
            Run a single episode of the Value Iteration Algorithm.

            Use:
                self.env: OpenAI env. env.P represents the transition probabilities of the environment.
                    env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
                    env.nS is a number of states in the environment.
                    env.nA is a number of actions in the environment.
                self.options.gamma: Gamma discount factor.
            """

        # Update each state...
        theta = 1e-5 #Initializing a very small of theta
        delta = 0 #Initializing the value of delta to be zero
        for s in range(self.env.nS): #looping through all the states
            # Do a one-step lookahead to find the best action
            # Update the value function. Ref: Sutton book eq. 4.10.

            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            # Do a one-step lookahead to calculate state-action values
                action_values = np.zeros(self.env.nA) #Initaizing all the action values of states
                for a in range(self.env.nA):
                     for prob, next_state, reward, done in self.env.P[s][a]:
                         action_values[a] += prob * (reward + self.options.gamma * self.V[next_state]) #updating the action values
                # Select best action to perform based on the highest state-action value
                best_action_value = np.max(action_values)
                # Calculate change in value
                delta = max(delta, np.abs(self.V[s] - best_action_value)) #updating the delta value
                # Update the value function for current state
                self.V[s] = best_action_value

        # In DP methods we don't interact with the environment so we will set the reward to be the sum of state values
        # and the number of steps to -1 representing an invalid value
        self.statistics[Statistics.Rewards.value] = np.sum(self.V)
        self.statistics[Statistics.Steps.value] = -1

    def __str__(self):
        return "Value Iteration"

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on state values.

        Use:
            self.env.nA: Number of actions in the environment.

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """

        def policy_fn(state):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            policy = np.zeros([self.env.nS, self.env.nA]) #initializing the 2D array
            for s in range(self.env.nS):
                # Do a one-step lookahead to calculate state-action values
                action_values = np.zeros(self.env.nA)
                for a in range(self.env.nA): #looping through list of all actions
                    for prob, next_state, reward, done in self.env.P[s][a]: #looping through list of transition tuples
                        action_values[a] += prob * (reward + self.options.gamma * self.V[next_state]) #list of all action values for the set of transition tuples
                # Select best action based on the highest state-action value
                best_action = np.argmax(action_values) #finding the best action value
                return best_action
        return policy_fn #return the best policy


class AsynchVI(ValueIteration):

    def __init__(self,env,options):
        super().__init__(env,options)
        # list of States to be updated by priority
        self.pq = PriorityQueue()
        # A mapping from each state to all states potentially leading to it in a single step
        self.pred = {}
        for s in range(self.env.nS):
            # Do a one-step lookahead to find the best action
            A = self.one_step_lookahead(s)
            best_action_value = np.max(A)
            self.pq.push(s, -abs(self.V[s]-best_action_value))
            for a in range(self.env.nA):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    if prob > 0:
                        if next_state not in self.pred.keys():
                            self.pred[next_state] = set()
                        if s not in self.pred[next_state]:
                            try:
                                self.pred[next_state].add(s)
                            except KeyError:
                                self.pred[next_state] = set()

    def train_episode(self):
        """
        Run a *single* update for Asynchronous Value Iteration Algorithm (using prioritized sweeping).
        Update only one state, the one with the highest priority

        Use:
            self.env: OpenAI env. env.P represents the transition probabilities of the environment.
                env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
                env.nS is a number of states in the environment.
                env.nA is a number of actions in the environment.
            self.options.gamma: Gamma discount factor.
            self.pred[s]: a list of states leading to state s in one step with probability > 0
            self.pq: list of States to be updated by priority
        """

        #########################################################
        # YOUR IMPLEMENTATION HERE                              #
        # Choose state with the maximal value change potential  #
        # Do a one-step lookahead to find the best action       #
        # Update the value function. Ref: Sutton book eq. 4.10. #
        #########################################################
        last_state = self.pq.pop() #popping the last state from the list of states
        list_of_actions = self.one_step_lookahead(last_state) #finding the list of all possible actions for the last state
        best_action = np.max(list_of_actions) #selecting the best action from the list of actions for the last state
        self.V[last_state] = best_action
        for state in self.pred[last_state]:
            list_of_actions = self.one_step_lookahead(state)
            best_action = np.max(list_of_actions)
            delta = abs(self.V[state]-best_action) #updating the value of delta
            self.pq.update(state,-delta) #updating the priority queue

        # In DP methods we don't interact with the environment so we will set the reward to be the sum of state values
        # and the number of steps to -1 representing an invalid value
        self.statistics[Statistics.Rewards.value] = np.sum(self.V)
        self.statistics[Statistics.Steps.value] = -1

    def __str__(self):
        return "Asynchronous VI"

    def one_step_lookahead(self,state):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(self.env.nA)
        for a in range(self.env.nA):
            for prob, next_state, reward, done in self.env.P[state][a]:
                A[a] += prob * (reward + self.options.gamma * self.V[next_state])
        return A


class PriorityQueue:
    """
    Implements a priority queue data structure. Each inserted item
    has a priority associated with it and the client is usually interested
    in quick retrieval of the lowest-priority item in the queue. This
    data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)
