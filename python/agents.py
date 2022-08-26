import numpy as np
from distribution import Distribution, DistributionArray

class ClassicalAgent():
    def __init__(self, env, gamma):
        raise Exception("Not implemented yet")

class DistributionalAgent():
    """
    Implements agents used for Distributionnal Reinforcement Learning. 
    It allows for easy use of Bellman and projections operators.

    Attributes:
        space_shape: the dimension of the space of states of the environment
        action_shape: the dimension of the space of the action
        transition_fun: the transition function of the environement
        
        gamma: the discount coefficient
        
        projection_type: the type of projection to use after each application of bellman operators
        resolution: the number of atom used in the projections.
        vmin, vmax: the bounds of reward for the categorical projection
    """
    def __init__(self, space_shape, action_shape, transition_fun, gamma=1, projection_type="None", resolution=100, vmin=None, vmax=None):
        self.gamma = gamma
        self.action_shape = action_shape
        self.state_shape = space_shape
        self.projection_type = projection_type
        self.resolution = resolution
        self.vmin, self.vmax = vmin, vmax

        self.transition_fun = transition_fun
        
        self.distribution_array = None

    def initialize(self):
        """Creates/Resets the array of distribution to represent each state and action"""
        self.distribution_array = DistributionArray((*self.state_shape, self.action_shape))

    def _project(self, projection_type=None):
        """Applie one of the following projection :
                -quantile : http://arxiv.org/abs/1710.10044
                -categorical : http://arxiv.org/abs/1707.06887
                -expectile : http://arxiv.org/abs/2112.15430
        """
        if projection_type == None:
            projection_type = self.projection_type

        if projection_type == "quantile":
            self.distribution_array.project_quantile(self.resolution)
        elif projection_type == "categorical":
            self.distribution_array.project_categorical(self.resolution, self.vmin, self.vmax)
        elif projection_type == "expectile":
            raise Exception("Not implemented yet")

    def evaluate(self, policy, projection_type=None):
        """Applies one step of the bellman operator, then projects the distribution.
        
        The policy should be a function of the form (state) -> (action)
        """
        new_darray = DistributionArray((*self.state_shape, self.action_shape), dic={})

        #Bellman operator
        for (*state, action) in np.ndindex((*self.state_shape, self.action_shape)):
            transitions = self.transition_fun(state, action)
            for (proba, new_state, reward) in transitions:
                new_darray[(*state, action)] = new_darray[(*state, action)] + (proba*(self.distribution_array[(*new_state, policy(new_state))].transfer(self.gamma, reward)))
        self.distribution_array = new_darray

        #Projection
        self._project(projection_type)

        return self.distribution_array
    
    def control(self, criterion, projection_type=None):
        """Applies one step of the optimal Bellman operator, then projects the distribution
        
        The criterion should be of the form (distribution1, distribution2) -> (bool)
        """
        new_darray = DistributionArray((*self.state_shape, self.action_shape), dic={})

        #Bellman operator
        for (*state, action) in np.ndindex((*self.state_shape, self.action_shape)):
            transitions = self.transition_fun(state, action)
            for (proba, new_state, reward) in transitions:
                optimal_action = self.get_optimal_action(new_state, criterion)
                new_darray[(*state, action)] = new_darray[(*state, action)] + (proba*(self.distribution_array[(*new_state, optimal_action)].transfer(self.gamma, reward)))
        self.distribution_array = new_darray

        #Projection
        self._project(projection_type)

        return self.distribution_array          

    def get_optimal_action(self, pos, criterion):
        """Returns the optimal action at state (pos) under (criterion)"""
        best_action = 0

        for action in range(self.action_shape):
            if criterion(self.distribution_array[(*pos,action)], self.distribution_array[(*pos,best_action)]):
                best_action = action
        
        return best_action

    def optimal_actions(self, criterion):
        """Returns the array of optimal actions for each states under (criterion)"""
        res = np.zeros(self.state_shape, dtype=int)

        for state in np.ndindex(self.state_shape):
            res[state] = self.get_optimal_action(state, criterion)

        return res
    
    def plot(self, state, action, threshold=0.00001, title=None, figsize=(15, 5)):
        """Plots the distribution of reward expected when doing (action) at state (state)"""
        index = (*state, action)
        distribution_copy = self.distribution_array[index]._copy()
        distribution_copy._clean(threshold)
        distribution_copy.plot(title, figsize)

    def get_distribution(self, state, action):
        """Returns the distribution associated to (state) and (action)"""
        index = (*state, action)
        return self.distribution_array[index]

