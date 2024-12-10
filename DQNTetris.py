import pygame
import numpy as np
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from collections import deque
import pickle
import os

# Constants
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BLOCK_SIZE = 40
FPS = 5
NUM_EPISODES = 10
REPLAY_MEMORY_SIZE = 2000
BATCH_SIZE = 4
GAMMA = 0.99
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
TARGET_UPDATE_FREQUENCY = 10

# File paths for saving/loading
MODEL_WEIGHTS_PATH = 'dqn_weights.h5'
REPLAY_BUFFER_PATH = 'replay_buffer.pkl'

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((BOARD_WIDTH * BLOCK_SIZE, BOARD_HEIGHT * BLOCK_SIZE))
pygame.display.set_caption("Tetris AI")

# Define colors
COLORS = {
    'I': (0, 255, 255),
    'J': (0, 0, 255),
    'L': (255, 165, 0),
    'O': (255, 255, 0),
    'S': (0, 255, 0),
    'T': (128, 0, 128),
    'Z': (255, 0, 0),
    'BACKGROUND': (0, 0, 0)
}

# Define Tetris pieces
SHAPES = {
    'I': [[1, 1, 1, 1]],
    'J': [[0, 0, 1], [1, 1, 1]],
    'L': [[1, 0, 0], [1, 1, 1]],
    'O': [[1, 1], [1, 1]],
    'S': [[0, 1, 1], [1, 1, 0]],
    'T': [[0, 1, 0], [1, 1, 1]],
    'Z': [[1, 1, 0], [0, 1, 1]]
}

# Game class (same as before)
class Tetris:
    # Same code as before
    pass

# DQN Model (same as before)
def build_dqn(input_shape, action_size):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(action_size, activation='linear'))

    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    return model

# Experience Replay Buffer (same as before)
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

    # Save the buffer to a file
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)

    # Load the buffer from a file
    def load(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.buffer = pickle.load(f)

# Training function with Save/Load functionality
def main():
    clock = pygame.time.Clock()
    tetris = Tetris()
    input_shape = (BOARD_HEIGHT, BOARD_WIDTH, 1)
    action_size = 4  # 4 actions: rotate, move left, move right, move down

    # Load the DQN model if exists
    dqn = build_dqn(input_shape, action_size)
    target_dqn = build_dqn(input_shape, action_size)
    
    if os.path.exists(MODEL_WEIGHTS_PATH):
        dqn.load_weights(MODEL_WEIGHTS_PATH)
        target_dqn.load_weights(MODEL_WEIGHTS_PATH)

    # Initialize the replay buffer and load it if it exists
    replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)
    replay_buffer.load(REPLAY_BUFFER_PATH)

    epsilon = 1.0

    for episode in range(NUM_EPISODES):
        tetris = Tetris()
        total_reward = 0
        done = False

        while not done:
            state = tetris.get_state()
            if np.random.rand() <= epsilon:
                action = np.random.choice(action_size)  # Explore
            else:
                q_values = dqn.predict(state.reshape(1, BOARD_HEIGHT, BOARD_WIDTH, 1))
                action = np.argmax(q_values)  # Exploit

            # Execute action
            if action == 0:  # Rotate
                tetris.rotate()
            elif action == 1:  # Move Left
                tetris.move_left()
            elif action == 2:  # Move Right
                tetris.move_right()
            elif action == 3:  # Move Down
                done = not tetris.move_down()

            # Get reward and next state
            reward = tetris.get_reward()
            next_state = tetris.get_state()
            done = tetris.game_over

            # Store in replay buffer
            replay_buffer.add((state, action, reward, next_state, done))

            # Sample minibatch from replay buffer
            if replay_buffer.size() > BATCH_SIZE:
                minibatch = replay_buffer.sample(BATCH_SIZE)
                for state_batch, action_batch, reward_batch, next_state_batch, done_batch in minibatch:
                    target = reward_batch
                    if not done_batch:
                        next_q_values = target_dqn.predict(next_state_batch.reshape(1, BOARD_HEIGHT, BOARD_WIDTH, 1))
                        target = reward_batch + GAMMA * np.max(next_q_values)

                    target_q_values = dqn.predict(state_batch.reshape(1, BOARD_HEIGHT, BOARD_WIDTH, 1))
                    target_q_values[0][action_batch] = target

                    dqn.fit(state_batch.reshape(1, BOARD_HEIGHT, BOARD_WIDTH, 1), target_q_values, epochs=1, verbose=0)

            # Update target network
            if episode % TARGET_UPDATE_FREQUENCY == 0:
                target_dqn.set_weights(dqn.get_weights())

            total_reward += reward

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY

            # Render the game state
            tetris.draw_board()

            # Print rewards for each step
            print(f"Step Reward: {reward}, Total Reward: {total_reward}")
            clock.tick(FPS)

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

        # Save the model weights and replay buffer after each episode
        dqn.save_weights(MODEL_WEIGHTS_PATH)
        replay_buffer.save(REPLAY_BUFFER_PATH)

if __name__ == "__main__":
    main()
