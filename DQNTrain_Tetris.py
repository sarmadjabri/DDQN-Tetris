import pygame
import numpy as np
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from collections import deque
import os
import time

# Constants
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BLOCK_SIZE = 40
FPS = 10
NUM_EPISODES = 500
REPLAY_MEMORY_SIZE = 2000
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
TARGET_UPDATE_FREQUENCY = 10
MODEL_PATH = "tetris_model.h5"

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
    'BACKGROUND': (0, 0, 0),
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

# Game class
class Tetris:
    def __init__(self):
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH))
        self.current_piece = self.new_piece()
        self.next_piece = self.new_piece()
        self.x = BOARD_WIDTH // 2 - len(self.current_piece[0]) // 2
        self.y = 0
        self.game_over = False
        self.score = 0
        self.last_action_time = time.time()

    def new_piece(self):
        shape = random.choice(list(SHAPES.keys()))
        return SHAPES[shape]

    def rotate(self):
        self.current_piece = np.rot90(self.current_piece)

    def move_left(self):
        self.x -= 1
        if self.check_collision():
            self.x += 1

    def move_right(self):
        self.x += 1
        if self.check_collision():
            self.x -= 1

    def move_down(self):
        self.y += 1
        if self.check_collision():
            self.y -= 1
            self.place_piece()
            return False
        return True

    def check_collision(self):
        for iy, row in enumerate(self.current_piece):
            for ix, cell in enumerate(row):
                if cell:
                    if (self.x + ix < 0 or self.x + ix >= BOARD_WIDTH or
                            self.y + iy >= BOARD_HEIGHT or self.board[self.y + iy][self.x + ix] != 0):
                        return True
        return False

    def place_piece(self):
        for iy, row in enumerate(self.current_piece):
            for ix, cell in enumerate(row):
                if cell:
                    if 0 <= self.y + iy < BOARD_HEIGHT and 0 <= self.x + ix < BOARD_WIDTH:
                        self.board[self.y + iy][self.x + ix] = 1
                    else:
                        self.game_over = True
                        return
        self.clear_lines()
        self.current_piece = self.next_piece
        self.next_piece = self.new_piece()
        self.x = BOARD_WIDTH // 2 - len(self.current_piece[0]) // 2
        self.y = 0
        if self.check_collision():
            self.game_over = True

    def clear_lines(self):
        lines_cleared = 0
        for i in range(BOARD_HEIGHT):
            if all(self.board[i]):
                self.board[i] = np.zeros(BOARD_WIDTH)
                lines_cleared += 1
        self.score += lines_cleared * 100

    def calculate_neatness(self):
        holes = 0
        for x in range(BOARD_WIDTH):
            found_block = False
            for y in range(BOARD_HEIGHT):
                if self.board[y][x] == 1:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes

    def stack_height(self):
        height = 0
        for y in range(BOARD_HEIGHT):
            if any(self.board[y]):
                height = BOARD_HEIGHT - y
                break
        return height

    def smoothness(self):
        smooth = 0
        for y in range(1, BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if self.board[y][x] == 1 and self.board[y - 1][x] == 0:
                    smooth += 1
        return smooth

    def get_state(self):
        return np.expand_dims(self.board, axis=-1)

    def draw_board(self):
        screen.fill(COLORS['BACKGROUND'])

        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                color = COLORS['BACKGROUND'] if self.board[y][x] == 0 else COLORS['I']
                pygame.draw.rect(screen, color, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        for iy, row in enumerate(self.current_piece):
            for ix, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(screen, COLORS['I'], ((self.x + ix) * BLOCK_SIZE, (self.y + iy) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        pygame.display.flip()

    def get_reward(self):
        reward = 0

        # Reward for clearing lines (especially Tetrises)
        lines_cleared = 0
        for i in range(BOARD_HEIGHT):
            if all(self.board[i]):
                lines_cleared += 1
        if lines_cleared > 0:
            if lines_cleared == 4:
                reward += 1500  # Tetris (clear 4 lines) gets a big bonus
            else:
                reward += 200 * lines_cleared  # Smaller bonus for fewer lines

        # Penalize for holes (empty spaces with blocks above them)
        holes = self.calculate_neatness()
        reward -= holes * 4  # Heavier penalty for holes

        # Penalize for stack height (taller stacks are worse)
        stack_height = self.stack_height()
        if stack_height > 15:
            reward -= (stack_height - 15) * 4  # Stronger penalty if stack is too high

        # Penalize for stack smoothness (jagged edges are worse)
        smoothness_penalty = self.smoothness()
        reward -= smoothness_penalty * 2  # Increased penalty for rough stacks

        return reward


# DQN Model
def build_dqn(input_shape, action_size):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))  # Adjusted to 64 filters
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(action_size, activation='linear'))

    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    return model


# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size ):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# Double DQN training loop
def main():
    episode_rewards = []

    clock = pygame.time.Clock()
    tetris = Tetris()
    input_shape = (BOARD_HEIGHT, BOARD_WIDTH, 1)
    action_size = 4  # 4 actions: rotate, move left, move right, move down

    dqn = build_dqn(input_shape, action_size)
    target_dqn = build_dqn(input_shape, action_size)
    target_dqn.set_weights(dqn.get_weights())

    replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)
    epsilon = 1.0  # Initial epsilon value for epsilon-greedy policy
    epsilon_min = MIN_EPSILON
    epsilon_decay = EPSILON_DECAY
    gamma = GAMMA  # Discount factor

    for episode in range(NUM_EPISODES):
        total_reward = 0
        tetris = Tetris()  # Reset game at the start of each episode

        while not tetris.game_over:
            state = tetris.get_state()
            state = np.expand_dims(state, axis=0)  # Add batch dimension

            if random.random() < epsilon:
                action = random.randint(0, action_size - 1)  # Random action
            else:
                q_values = dqn.predict(state)
                action = np.argmax(q_values)  # Action with highest Q-value

            # Take the chosen action
            if action == 0:
                tetris.rotate()
            elif action == 1:
                tetris.move_left()
            elif action == 2:
                tetris.move_right()
            elif action == 3:
                tetris.move_down()

            reward = tetris.get_reward()
            total_reward += reward

            next_state = tetris.get_state()
            next_state = np.expand_dims(next_state, axis=0)

            replay_buffer.add((state, action, reward, next_state, tetris.game_over))

            if replay_buffer.size() > BATCH_SIZE:
                batch = replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                # Prepare data for training
                states = np.vstack(states)
                next_states = np.vstack(next_states)
                actions = np.array(actions)
                rewards = np.array(rewards)
                dones = np.array(dones)

                q_values = dqn.predict(states)
                next_q_values = target_dqn.predict(next_states)

                for i in range(BATCH_SIZE):
                    if dones[i]:
                        q_values[i, actions[i]] = rewards[i]
                    else:
                        q_values[i, actions[i]] = rewards[i] + gamma * np.max(next_q_values[i])

                dqn.fit(states, q_values, epochs=1, verbose=0)

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            if episode % TARGET_UPDATE_FREQUENCY == 0:
                target_dqn.set_weights(dqn.get_weights())

            # Print step-level details
            print(f"Episode: {episode + 1}, Step Reward: {reward}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

        print(f"Episode {episode + 1} finished. Total Reward: {total_reward}")

    # Model saving after all episodes
    if episode % 10 == 0:
        print("Saving model...")
        dqn.save_weights(MODEL_PATH)


if __name__ == "__main__":
    main()
