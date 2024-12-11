import pygame
import numpy as np
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from collections import deque
import os
import matplotlib.pyplot as plt

# Constants
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BLOCK_SIZE = 40  # Increased block size for better visualization
FPS = 60
NUM_EPISODES = 1000
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
                    self.board[self.y + iy][self.x + ix] = 1
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
        return -holes  # Negative reward for holes

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
        return -smooth

    def get_state(self):
        return np.expand_dims(self.board, axis=-1)

    def draw_board(self):
        screen.fill(COLORS['BACKGROUND'])

        # Draw the filled blocks on the board
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                color = COLORS['BACKGROUND'] if self.board[y][x] == 0 else COLORS['I']
                pygame.draw.rect(screen, color, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        # Draw the falling piece
        for iy, row in enumerate(self.current_piece):
            for ix, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(screen, COLORS['I'], ((self.x + ix) * BLOCK_SIZE, (self.y + iy) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        pygame.display.flip()

    def get_reward(self):
        reward = 0

        # Reward for line clearing
        lines_cleared = 0
        for i in range(BOARD_HEIGHT):
            if all(self.board[i]):
                lines_cleared += 1
        if lines_cleared > 0:
            reward += 100 * lines_cleared  # High reward for clearing multiple lines

        # Penalize for holes in the board
        reward += self.calculate_neatness()  # Negative for more holes

        # Reward for smooth stack (fewer peaks/valleys)
        reward += self.smoothness()

        # Reward for lower stack height
        reward -= self.stack_height() * 0.1

        # Penalize for game over
        if self.game_over:
            reward -= 500  # Large penalty for game over

        return reward


# DQN Model
def build_dqn(input_shape, action_size):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(action_size, activation='linear'))

    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    return model


# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# Training function with Double DQN
def plot_metrics(episode_rewards):
    # Plot total reward per episode
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward per Episode')
    plt.show()


def main():
    # List to store rewards for each episode
    episode_rewards = []

    clock = pygame.time.Clock()
    tetris = Tetris()
    input_shape = (BOARD_HEIGHT, BOARD_WIDTH, 1)
    action_size = 4  # 4 actions: rotate, move left, move right, move down

    dqn = build_dqn(input_shape, action_size)
    target_dqn = build_dqn(input_shape, action_size)
    target_dqn.set_weights(dqn.get_weights())

    replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)
    epsilon = 1.0

    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        dqn.load_weights(MODEL_PATH)
        target_dqn.set_weights(dqn.get_weights())

    for episode in range(NUM_EPISODES):
        tetris = Tetris()
        total_reward = 0
        done = False

        while not done:
            state = tetris.get_state()
            if random.random() < epsilon:
                action = random.randint(0, action_size - 1)  # Random action (exploration)
            else:
                action = np.argmax(dqn.predict(state))  # Choose the best action (exploitation)

            if action == 0:
                tetris.rotate()
            elif action == 1:
                tetris.move_left()
            elif action == 2:
                tetris.move_right()
            elif action == 3:
                done = not tetris.move_down()

            reward = tetris.get_reward()
            next_state = tetris.get_state()
            replay_buffer.add((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            if len(replay_buffer.buffer) >= BATCH_SIZE:
                minibatch = replay_buffer.sample(BATCH_SIZE)
                for s, a, r, s_next, done in minibatch:
                    target = r
                    if not done:
                        target += GAMMA * np.max(target_dqn.predict(s_next))

                    target_f = dqn.predict(s)
                    target_f[0][a] = target
                    dqn.fit(s, target_f, epochs=1, verbose=0)

                if episode % TARGET_UPDATE_FREQUENCY == 0:
                    target_dqn.set_weights(dqn.get_weights())

        # Track the progress
        print(f"Episode {episode + 1}/{NUM_EPISODES} | Total Reward: {total_reward}")

        episode_rewards.append(total_reward)

        # Epsilon decay
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY

        # Save the model periodically
        if (episode + 1) % 50 == 0:
            dqn.save_weights(MODEL_PATH)
            print(f"Model saved after episode {episode + 1}")

    # Plot the rewards graph after training
    plot_metrics(episode_rewards)


if __name__ == "__main__":
    main()
