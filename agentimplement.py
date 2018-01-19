import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here

        #Storing Q-values for states & actions in a dictionary
        self.Qtable = dict()
        
        self.alpha = 0.9   #Learning rate
        self.gamma = 0.2   #Discount factor for value of future reward
        self.epsilon = 0.05  #Exploration rate
        
        self.possible_actions = Environment.valid_actions
        self.action = None
        self.reward = None
        self.totalreward = 0
        self.steps = 0
                
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.next_waypoint
        self.steps = 0
        self.totalreward = 0

    def qVal(self, state, action):
        key = (state, action)
        return self.Qtable.get(key, 4.0)
    
    def maxQ(self, state):
        q = [self.qVal(state,a) for a in self.possible_actions]
        return max(q)
        
    def nextAction(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.possible_actions)
        else:
            q = [self.qVal(state,a) for a in self.possible_actions]
            if q.count(max(q)) > 1:  #if there are multiple actions with best q
                bestaction = [i for i in range(len(self.possible_actions)) if q[i] == max(q)]
                index = random.choice(bestaction)
            else:
                index = q.index(max(q))
            action = self.possible_actions[index]
        return action
        
    def qLearn(self, state, action, nextState, reward):
        key = (state, action)
        if (key not in self.Qtable):
            self.Qtable[key] = 4  #set initial q
        else:
            self.Qtable[key] = self.Qtable[key] + self.alpha*(reward + self.gamma*self.maxQ(nextState) - self.Qtable[key])
              
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.new_state = inputs
        self.new_state['next_waypoint'] = self.next_waypoint
        self.new_state = tuple(sorted(self.new_state.iteritems()))
                
        # TODO: Select action according to your policy
        action = self.nextAction(self.new_state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        if self.reward != None:
            self.qLearn(self.state, self.action, self.new_state, self.reward)

        self.action = action
        self.state = self.new_state
        self.reward = reward
        self.totalreward = self.totalreward + reward
        self.steps = self.steps + 1

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
