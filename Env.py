import numpy as np
import math
import random
from itertools import permutations

m = 5  # number of cities
t = 24  # number of hours
d = 7  # number of days
C = 5  # Per hour fuel and other costs
R = 9  # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        self.action_space = [(0, 0)] + list(permutations([i for i in range(m)], 2))
        self.state_space = [[x, y, z] for x in range(m) for y in range(t) for z in range(d)]
        self.state_init = random.choice(self.state_space)
        self.reset()

    def state_encod_arch1(self, state):
        state_encod = [0 for _ in range(m+t+d)]
        state_encod[self.state_get_loc(state)] = 1
        state_encod[m+self.state_get_time(state)] = 1
        state_encod[m+t+self.state_get_day(state)] = 1

        return state_encod

    def requests(self, state):
        """Number of requests based on location. Used the table specified in MDP"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)

        if requests > 15:
            requests = 15
        possible_actions_index = random.sample(range(1, (m-1)*m + 1), requests) + [0]
        actions = [self.action_space[i] for i in possible_actions_index]

        return possible_actions_index, actions

    def update_time_day(self, time, day, ride_duration):
        ride_duration = int(ride_duration)

        if (time + ride_duration) < 24:
            time = time + ride_duration
        else:
            # convert the time to 0-23 range
            time = (time + ride_duration) % 24 
            
            # Get the number of days
            num_days = (time + ride_duration) // 24
            
            # Convert the day to 0-6 range
            day = (day + num_days ) % 7

        return time, day
    
    def next_state_func(self, state, action, Time_matrix):
        """Return next state"""
        next_state = []
        
        # Initialize various times
        total_time   = 0
        transit_time = 0
        wait_time    = 0   
        ride_time    = 0    
        
        # Derive the current location, time, day and request locations
        curr_loc = self.state_get_loc(state)
        pickup_loc = self.action_get_pickup(action)
        drop_loc = self.action_get_drop(action)
        curr_time = self.state_get_time(state)
        curr_day = self.state_get_day(state)
           
        if ((pickup_loc== 0) and (drop_loc == 0)):
            # Refuse all requests, so wait time is 1 unit, next location is current location
            wait_time = 1
            next_loc = curr_loc
        elif (curr_loc == pickup_loc):
            # means driver is already at pickup point, wait and transit are both 0 then.
            ride_time = Time_matrix[curr_loc][drop_loc][curr_time][curr_day]
            
            # next location is the drop location
            next_loc = drop_loc
        else:
            # time take to reach pickup point
            transit_time      = Time_matrix[curr_loc][pickup_loc][curr_time][curr_day]
            new_time, new_day = self.update_time_day(curr_time, curr_day, transit_time)
            
            # The driver is now at the pickup point
            ride_time = Time_matrix[pickup_loc][drop_loc][new_time][new_day]
            next_loc  = drop_loc

        total_time = (wait_time + transit_time + ride_time)
        next_time, next_day = self.update_time_day(curr_time, curr_day, total_time)
        
        next_state = [next_loc, next_time, next_day]
        
        return next_state, wait_time, transit_time, ride_time
    

    def reset(self):
        return self.action_space, self.state_space, self.state_init

    def reward_func(self, wait_time, transit_time, ride_time):
        passenger_time = ride_time
        idle_time      = wait_time + transit_time
        
        reward = (R * passenger_time) - (C * (passenger_time + idle_time))

        return reward

    def step(self, state, action, Time_matrix):
        next_state, wait_time, transit_time, ride_time = self.next_state_func(
            state, action, Time_matrix)

        # Calculating reward based on time and location
        rewards = self.reward_func(wait_time, transit_time, ride_time)
        total_time = wait_time + transit_time + ride_time
        
        return rewards, next_state, total_time

    def state_get_loc(self, state):
        return state[0]

    def state_get_time(self, state):
        return state[1]

    def state_get_day(self, state):
        return state[2]

    def action_get_pickup(self, action):
        return action[0]

    def action_get_drop(self, action):
        return action[1]

    def state_set_loc(self, state, loc):
        state[0] = loc

    def state_set_time(self, state, time):
        state[1] = time

    def state_set_day(self, state, day):
        state[2] = day

    def action_set_pickup(self, action, pickup):
        action[0] = pickup

    def action_set_drop(self, action, drop):
        action[1] = drop
