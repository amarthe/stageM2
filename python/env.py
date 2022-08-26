import enum
import numpy as np
from distribution import Distribution
import abc

#########
# Utils #
#########

class Action(enum.Enum):
    """Inplements the 4 directions"""
    Down = 0
    Up = 1
    Left = 2
    Right = 3

def action_to_dx(action):
    """given a action, returns the coordinate move associated"""
    if action.name == "Down":
        return (0,-1)
    if action.name == "Up":
        return (0,1)
    if action.name == "Left":
        return (-1,0)
    if action.name == "Right":
        return (1,0)
    else:
        raise Exception("Error in action_to_dx")

def pair_sum(pair1, pair2, p_min, p_max):
    """util function to perform addition on pairs and clip them"""
    return(np.clip(pair1[0] + pair2[0], p_min[0], p_max[0]), np.clip(pair1[1] + pair2[1], p_min[1], p_max[1]))


class DistributionalEnv(metaclass=abc.ABCMeta):
    """Base class for environment used in Distributional RL"""

    @property
    @abc.abstractmethod
    def transition_function(self):
        """
        returns the transition function with the following specs:
            Input:
                pos: the position of the agent
                action: the action to perform
            Output:
                a list of transitions of the form [proba, state_reached, reward_obtained]
        """
        pass

    @property
    @abc.abstractmethod
    def space_shape():
        pass

    @property
    @abc.abstractmethod
    def action_shape():
        pass


####################################
#            Environments          #
####################################

class Cliff(DistributionalEnv):
    """
    Implements the cliff environment similar to the one in Sutton & Barto,
    to test with RL algorithms. It looks like :

        C C C C C 
        C C C C C
        C C C C C
        B X X X E

    B is where the environment starts, E where it ends. C are the regular cells, and X the cliff.
    The agent has to go from B to E without falling in X.
    At each step, the agent has 4 possible actions, the 4 direction. For every action, the agent
    has 0.7 to go it that direction, and 0.1 in each other directions. A action leading to out of
    bound makes the agent stay on the same place.

    Attributes:
        nb_col: number of columns of the environement
        nb_line: number of lines of the environment

        reward_step: the rewards given at each step (except cliff and end)
        reward_end: the reward given when reaching the end
        reward_fall: the reward given when falling off the cliff

        space_shape: (nb_col, nb_line)
        action_shape: the number of actions

        self.x: x-pos of the agent
        self.y: y-pos of the agent
    """

    def __init__(self,  nb_col=5, nb_line=4, reward_step=0, reward_end=10, reward_fall=-10):

        self.nb_col = nb_col
        self.nb_line = nb_line
        self.reward_step = reward_step
        self.reward_end = reward_end
        self.reward_fall = reward_fall

        self._space_shape = (nb_col, nb_line)
        self._action_shape = (4)

        self.x = 0
        self.y = 0
        self.end = (nb_col-1,0)

        self.dspace = None

    @property
    def space_shape(self):
        return self._space_shape   
    
    @property
    def action_shape(self):
        return self._action_shape

    def _sto_transition(self, pos, action):
        """an in-class method of the transition function"""
        x, y = pos
        trans = [[0.1, pair_sum((x,y), action_to_dx(act), (0,0), (self.nb_col-1, self.nb_line-1)), self.reward_step] for act in list(Action)]
        trans[action][0] = 0.7

        if x==0 and y==0:                      #bottom left corner
            trans[Action.Right.value] = [trans[Action.Right.value][0], (0,0), self.reward_fall]
            return trans
        elif 0 < x < self.nb_col-1 and y==1:   #along the cliff
            trans[Action.Down.value] = [trans[Action.Down.value][0], (0,0), self.reward_fall]
            return trans
        elif x==self.nb_col-1 and y==1:        #before the goal
            trans[Action.Down.value][2] = self.reward_end
            return trans
        elif x==self.nb_col-1 and y==0:        #on the goal
            return [[1,(x,y),0]]
        else:
            return trans         

    @property
    def transition_table(self):
        shape = (*self.space_shape, self.action_shape)
        trans_array = np.ndarray(shape, dtype=list)
        for *state, action in np.ndindex(shape):
            trans_array[(*state, action)] = self._sto_transition(state, action)
        
        return trans_array

    @property
    def transition_function(self):

        def fun(pos, action):
            x, y = pos
            trans = [[0.1, pair_sum((x,y), action_to_dx(act), (0,0), (self.nb_col-1, self.nb_line-1)), self.reward_step] for act in list(Action)]
            trans[action][0] = 0.7

            if x==0 and y==0:                      #bottom left corner
                trans[Action.Right.value] = [trans[Action.Right.value][0], (0,0), self.reward_fall]
                return trans
            elif 0 < x < self.nb_col-1 and y==1:   #along the cliff
                trans[Action.Down.value] = [trans[Action.Down.value][0], (0,0), self.reward_fall]
                return trans
            elif x==self.nb_col-1 and y==1:        #before the goal
                trans[Action.Down.value][2] = self.reward_end
                return trans
            elif x==self.nb_col-1 and y==0:        #on the goal
                return [[1,(x,y),0]]
            else:
                return trans  

        return fun
    
    def display_actions(self, action_array):
        action_symbols = ["↓","↑","←","→"]
        nb_col, nb_line = action_array.shape
        for y in reversed(range(nb_line)):
            for x in range(nb_col):
                print(action_symbols[action_array[x,y]], end=" ")
            print("")

    # def observation_spec(self):
    #     return specs.BoundedArray(
    #         shape=(2),
    #         dtype=int,
    #         name="space",
    #         minimum=(0,0),
    #         maximum=self.space_shape,
    #     )

    # def action_spec(self):
    #     return specs.DiscreteArray(
    #         dtype=int, 
    #         num_values=len(list(Action)), 
    #         name="action")
    
    # def _observe(self):
    #    """returns the positiona of the agent"""
    #    return (self.x, self.y)

    # def reset(self):
    #    raise Exception("Not implemented yet")

    # def step(self, action):
    #    "samples a transition and update the position"
    #    raise Exception("Not implemented yet")

class SimplestEnv(DistributionalEnv):
    "Environment with only one state and 2 actions"

    def __init__(self):
        self._space_shape = (1)
        self._action_shape = (2)

    @property
    def space_shape(self):
        return self._space_shape   
    
    @property
    def action_shape(self):
        return self._action_shape

    def transition_function(self):
        
        def fun(pos, action):
            if action == 0:
                return [[1,(0),1]]
            elif action == 1:
                return [[0.5,(0),0],[0.5,(0),2]]
            else:
                raise Exception("action out of range")
    
        return fun

class SimpleEnv(DistributionalEnv):
    "Environment  with only two states and 2 actions"

    def __init__(self):
        self._space_shape = (2)
        self._action_shape = (2)

    @property
    def space_shape(self):
        return self._space_shape   
    
    @property
    def action_shape(self):
        return self._action_shape
    
    def transition_function(self):
        
        def fun(pos, action):
            if action == 0 and pos == (0):
                return [[1,(0),0]]
            elif action == 1 and pos == (0):
                return [[0.5,(0),-1],[0.5,(1),1]]
            elif action == 0 and pos == (1):
                return [[1,(1),1]]
            elif action == 1 and pos == (1):
                return [[0.5,(0),2],[0.5,(1),0]]
            else:
                raise Exception("State or Action out of bound")

        return fun