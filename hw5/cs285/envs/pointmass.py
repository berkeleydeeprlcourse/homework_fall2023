import numpy as np
import gym
import pickle

WALLS = {
    'Small':
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]),
    'Cross':
        np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]]),
    'FourRooms':
        np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]),
    'Spiral5x5':
        np.array([[0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1],
                  [0, 1, 0, 0, 1],
                  [0, 1, 1, 0, 1],
                  [0, 0, 0, 0, 1]]),
    'Spiral7x7':
        np.array([[1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 0],
                  [1, 0, 1, 0, 0, 1, 0],
                  [1, 0, 1, 1, 0, 1, 0],
                  [1, 0, 0, 0, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 0]]),
    'Spiral9x9':
        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 1, 0, 0, 0, 0, 0, 0, 1],
                  [0, 1, 0, 1, 1, 1, 1, 0, 1],
                  [0, 1, 0, 1, 0, 0, 1, 0, 1],
                  [0, 1, 0, 1, 1, 0, 1, 0, 1],
                  [0, 1, 0, 0, 0, 0, 1, 0, 1],
                  [0, 1, 1, 1, 1, 1, 1, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1]]),
    'Spiral11x11':
        np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                  [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                  [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]),
    'Maze5x5':
        # np.array([[0, 0, 0],
        #           [1, 1, 0],
        #           [0, 0, 0]]),
        np.array([[0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 0],
                  [1, 1, 1, 1, 0],
                  [1, 1, 1, 1, 0],
                  [1, 1, 1, 1, 0]]),
    'Maze6x6':
        np.array([[0, 0, 1, 0, 0, 0],
                  [1, 0, 1, 0, 1, 0],
                  [0, 0, 1, 0, 1, 1],
                  [0, 1, 1, 0, 0, 1],
                  [0, 0, 1, 1, 0, 1],
                  [1, 0, 0, 0, 0, 1]]),
    'Maze11x11':
        np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                  [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    'Tunnel':
        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                  [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                  [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
                  [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]),
    'U':
        np.array([[0, 0, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [1, 1, 0],
                  [1, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 0, 0]]),
    'Tree':
        np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        ]),
    'UMulti':
        np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         ]),
    'FlyTrapSmall':
        np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         ]),
    'FlyTrapBig':
        np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         ]),
    'Galton':
        np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        ]),
}

ACT_DICT = {
    0: [0.0, 0.0],
    1: [0.0, -1.0],
    2: [0.0, 1.0],
    3: [-1.0, 0.0],
    4: [1.0, 0.0],
}


def resize_walls(walls, factor):
    """Increase the environment by rescaling.

    Args:
      walls: 0/1 array indicating obstacle locations.
      factor: (int) factor by which to rescale the environment."""
    (height, width) = walls.shape
    row_indices = np.array([i for i in range(height) for _ in range(factor)])
    col_indices = np.array([i for i in range(width) for _ in range(factor)])
    walls = walls[row_indices]
    walls = walls[:, col_indices]
    assert walls.shape == (factor * height, factor * width)
    return walls


class Pointmass(gym.Env):
    """Abstract class for 2D navigation environments."""

    def __init__(
        self,
        difficulty=0,
        dense_reward=False,
    ):
        """Initialize the point environment.

        Args:
          walls: (str) name of one of the maps defined above.
          resize_factor: (int) Scale the map by this factor.
          action_noise: (float) Standard deviation of noise to add to actions. Use 0
            to add no noise.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.plt = plt
        self.fig = self.plt.figure()

        self.action_dim = self.ac_dim = 2
        self.observation_dim = self.obs_dim = 2
        self.env_name = "pointmass"
        self.is_gym = True

        if difficulty == 0:
            walls = "Maze5x5"
            resize_factor = 2
            self.fixed_start = np.array([0.5, 0.5]) * resize_factor
            self.fixed_goal = np.array([4.5, 4.5]) * resize_factor
            self.max_episode_steps = 50
        elif difficulty == 1:
            walls = "Maze6x6"
            resize_factor = 1
            self.fixed_start = np.array([0.5, 0.5]) * resize_factor
            self.fixed_goal = np.array([1.5, 5.5]) * resize_factor
            self.max_episode_steps = 150
        elif difficulty == 2:
            walls = "FourRooms"
            resize_factor = 2
            self.fixed_start = np.array([1.0, 1.0]) * resize_factor
            self.fixed_goal = np.array([10.0, 10.0]) * resize_factor
            self.max_episode_steps = 100
        elif difficulty == 3:
            # NOTE TO STUDENTS: FEEL FREE TO EDIT THESE PARAMS FOR THE EXTRA CREDIT PROBLEM!
            walls = "Maze11x11"
            resize_factor = 1
            self.fixed_start = np.array([0.5, 0.5]) * resize_factor
            self.fixed_goal = np.array([0.5, 10.5]) * resize_factor
            self.max_episode_steps = 200
        else:
            print("Invalid difficulty setting")
            return 1 / 0

        if resize_factor > 1:
            self._walls = resize_walls(WALLS[walls], resize_factor)
        else:
            self._walls = WALLS[walls]
        (height, width) = self._walls.shape
        self._apsp = self._compute_apsp(self._walls)

        self._height = height
        self._width = width
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self._height, self._width]),
            dtype=np.float64,
        )

        self.dense_reward = dense_reward
        self.num_actions = 5
        self.epsilon = resize_factor
        self.action_noise = 0.5

        self.difficulty = difficulty

        self.num_runs = 0
        self.reset()

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, seed=None):
        if seed:
            self.seed(seed)

        self.plt.clf()
        self.timesteps_left = self.max_episode_steps

        self.state = self.fixed_start.copy()
        self.num_runs += 1
        return self._normalize_obs(self.state.copy())

    def _get_distance(self, obs, goal):
        """Compute the shortest path distance.

        Note: This distance is *not* used for training."""
        (i1, j1) = self._discretize_state(obs.copy())
        (i2, j2) = self._discretize_state(goal.copy())
        return self._apsp[i1, j1, i2, j2]

    def simulate_step(self, state, action):
        num_substeps = 10
        dt = 1.0 / num_substeps
        num_axis = len(action)
        for _ in np.linspace(0, 1, num_substeps):
            for axis in range(num_axis):
                new_state = state.copy()
                new_state[axis] += dt * action[axis]

                if not self._is_blocked(new_state):
                    state = new_state
        return state

    def get_optimal_action(self, state):
        state = self._unnormalize_obs(state)
        best_action = 0
        best_dist = np.inf
        for i in range(self.num_actions):
            action = np.array(ACT_DICT[i])
            s_prime = self.simulate_step(state, action)
            dist = self._get_distance(s_prime, self.fixed_goal)
            if dist < best_dist:
                best_dist = dist
                best_action = i
        return best_action

    def _discretize_state(self, state, resolution=1.0):
        (i, j) = np.floor(resolution * state).astype(np.int32)
        # Round down to the nearest cell if at the boundary.
        if i == self._height:
            i -= 1
        if j == self._width:
            j -= 1
        return (i, j)

    def _normalize_obs(self, obs):
        return np.array([obs[0] / float(self._height), obs[1] / float(self._width)])

    def _unnormalize_obs(self, obs):
        return np.array([obs[0] * float(self._height), obs[1] * float(self._width)])

    def _is_blocked(self, state):
        if not self.observation_space.contains(state):
            return True
        (i, j) = self._discretize_state(state)
        return self._walls[i, j] == 1

    def step(self, action):
        self.timesteps_left -= 1

        if isinstance(action, np.ndarray):
            action = action.item()

        action = np.array(ACT_DICT[action])
        action = np.random.normal(action, self.action_noise)
        self.state = self.simulate_step(self.state, action)

        dist = np.linalg.norm(self.state - self.fixed_goal)
        done = (dist < self.epsilon) or (self.timesteps_left == 0)
        ns = self._normalize_obs(self.state.copy())

        if self.dense_reward:
            reward = -dist
        else:
            reward = int(dist < self.epsilon) - 1

        return ns, reward, done, {}

    @property
    def walls(self):
        return self._walls

    @property
    def goal(self):
        return self._normalize_obs(self.fixed_goal.copy())

    def _compute_apsp(self, walls):
        import networkx as nx

        (height, width) = walls.shape
        g = nx.Graph()
        # Add all the nodes
        for i in range(height):
            for j in range(width):
                if walls[i, j] == 0:
                    g.add_node((i, j))

        # Add all the edges
        for i in range(height):
            for j in range(width):
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == dj == 0:
                            continue  # Don't add self loops
                        if i + di < 0 or i + di > height - 1:
                            continue  # No cell here
                        if j + dj < 0 or j + dj > width - 1:
                            continue  # No cell here
                        if walls[i, j] == 1:
                            continue  # Don't add edges to walls
                        if walls[i + di, j + dj] == 1:
                            continue  # Don't add edges to walls
                        g.add_edge((i, j), (i + di, j + dj))

        # dist[i, j, k, l] is path from (i, j) -> (k, l)
        dist = np.full((height, width, height, width), float("inf"))
        for (i1, j1), dist_dict in nx.shortest_path_length(g):
            for (i2, j2), d in dist_dict.items():
                dist[i1, j1, i2, j2] = d

        return dist

    def plot_trajectory(self, observations):
        fig = self.plt.figure()
        ax = fig.add_subplot(111)
        self.plot_walls(ax)

        ax.plot(observations[:, 0], observations[:, 1], "b-o", alpha=0.3)
        ax.scatter(
            [observations[0, 0]],
            [observations[0, 1]],
            marker="+",
            color="red",
            s=200,
            label="start",
        )
        ax.scatter(
            [observations[-1, 0]],
            [observations[-1, 1]],
            marker="+",
            color="green",
            s=200,
            label="end",
        )
        ax.scatter([self.goal[0]], [self.goal[1]], marker="*", color="green", s=200, label="goal")
        ax.legend(loc="upper left")
        return fig
    
    def plot_keypoints(self, ax):
        start = self._normalize_obs(self.fixed_start.copy())
        goal = self._normalize_obs(self.fixed_goal.copy())
        ax.scatter(
            [start[0]],
            [start[1]],
            marker="+",
            color="red",
            s=200,
            label="start",
        )
        ax.scatter([goal[0]], [goal[1]], marker="*", color="green", s=200, label="goal")

    def plot_walls(self, ax, walls=None):
        import matplotlib.pyplot as plt

        ax: plt.Axes = ax

        if walls is None:
            walls = self._walls.T
        (height, width) = walls.shape
        for i, j in zip(*np.where(walls)):
            x = np.array([j, j + 1]) / float(width)
            y0 = np.array([i, i]) / float(height)
            y1 = np.array([i + 1, i + 1]) / float(height)
            ax.fill_between(x, y0, y1, color="grey")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    def _sample_normalized_empty_state(self):
        s = self._sample_empty_state()
        return self._normalize_obs(s)

    def _sample_empty_state(self):
        candidate_states = np.where(self._walls == 0)
        num_candidate_states = len(candidate_states[0])
        state_index = np.random.choice(num_candidate_states)
        state = np.array(
            [candidate_states[0][state_index], candidate_states[1][state_index]],
            dtype=float,
        )
        state += np.random.uniform(size=2)
        assert not self._is_blocked(state)
        return state


def refresh_path():
    path = dict()
    path["observations"] = []
    path["actions"] = []
    path["next_observations"] = []
    path["terminals"] = []
    path["rewards"] = []
    return path


if __name__ == "__main__":
    env = Pointmass(difficulty=0, dense_reward=False)
    num_samples = 50000
    total_samples = 0
    path = refresh_path()
    all_paths = []
    num_positive_rewards = 0

    while total_samples < num_samples:
        path = refresh_path()
        start_state = env._sample_empty_state()
        bern = np.random.rand() > 0.5
        if bern:
            goal_state = env._sample_empty_state()
        else:
            goal_state = env.fixed_goal

        print("Start: ", start_state, " Goal state: ", goal_state, total_samples)
        # curr_state = start_state
        curr_state = env.reset(start_state)
        done = False
        for i in range(env.max_episode_steps):
            action = env.get_optimal_action(goal_state)
            temp_bern = np.random.rand() < 0.2
            if temp_bern:
                action = np.random.randint(5)

            next_state, reward, done, _ = env.step(action)
            if reward >= 0:
                num_positive_rewards += 1
            path["observations"].append(curr_state)
            path["actions"].append(action)
            path["next_observations"].append(next_state)
            path["terminals"].append(done)
            path["rewards"].append(reward)

            if done == True:
                total_samples += i
                break

        all_paths.append(path)
        print("Num Positive Rewards: ", num_positive_rewards)

    with open("buffer_debug_final" + str(env.difficulty) + ".pkl", "wb") as f:
        pickle.dump(all_paths, f)
