import numpy as np

from Solvers.Abstract_Solver import AbstractSolver, Statistics


class RandomWalk(AbstractSolver):
    def __init__(self,env,options):
        super().__init__(env,options)

    def train_episode(self):
        for t in range(self.options.steps):
            action = self.action_space.sample()
            observation, reward, done, info = self.env.step(action)
            if done:
                break
        print("Episode {} finished after {} timesteps with total rewards {}".format(
            self.statistics[Statistics.Episode.value],self.statistics[Statistics.Steps.value],
            self.statistics[Statistics.Rewards.value]))

    def __str__(self):
        return "Random Walk"

    def create_greedy_policy(self):
        """
        Creates a random policy function.

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities
        """
        nA = self.env.action_space.n
        A = np.ones(nA, dtype=float) / nA

        def policy_fn(observation):
            return A

        return policy_fn
