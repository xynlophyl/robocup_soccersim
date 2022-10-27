#!/usr/bin/env python

import threading
import time
import random
import sys
import multiprocessing as mp
import os

# import agent types (positions)
from agents.ball_agent import Agent as A0

test_agent = A0

# set team
TEAM_NAME = 'COMP423'
# NUM_PLAYERS = 11
NUM_PLAYERS = 10

# return type of agent: midfield, striker etc.
# NELSON: MOVED FUNCTIONS TO OUTSIDE MAIN CALL
def agent_type(position):
    return {
        2: test_agent,
        3: test_agent,
        4: test_agent,
        6: test_agent,
        7: test_agent,
        8: test_agent,
    }.get(position, test_agent)

# spawn an agent of team_name, with position
def spawn_agent(team_name, position):
    """
    Used to run an agent in a seperate physical process.
    """
    # return type of agent by position, construct
    a = agent_type(position)() # what is this?
    a.connect("localhost", 6000, team_name)
    a.play()

    # we wait until we're killed
    while 1:
        # we sleep for a good while since we can only exit if terminated.
        time.sleep(1)



if __name__ == "__main__":


    # spawn all agents as seperate processes for maximum processing efficiency
    agentthreads = []
    for position in range(1, NUM_PLAYERS+1):
    # for position in range(1):

        print(f"  Spawning agent {position}...")

        at = mp.Process(target=spawn_agent, args=(TEAM_NAME, position))
        # input()

        at.daemon = True
        at.start()

        agentthreads.append(at)

    print(f"Spawned {len(agentthreads)} agents.")
    print()
    print("Playing soccer...")
    # wait until killed to terminate agent processes
    try:
        while 1:
            time.sleep(0.05)
    except KeyboardInterrupt: 
        print()
        print("Killing agent threads...")

        # terminate all agent processes
        count = 0
        for at in agentthreads:
            print(f"  Terminating agent {count}...")
            at.terminate()
            count += 1
        print(f"Killed {count-1} agent threads.")

        print()
        print("Exiting.")
        sys.exit()
