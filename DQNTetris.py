import pygame
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from collections import deque

# Constants
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BLOCK_SIZE = 30
FPS = 60
NUM_EPISODES = 1000
REPLAY_MEMORY_SIZE = 2000
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

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
        self.score = 0
        self.game_over = False

    def new_piece(self):
        shape = random.choice(list(SHAPES.keys()))
        return SHAPES[shape]

    def clear_lines(self):
        lines_cleared = 0
        for i in range(BOARD_HEIGHT):
            if all(self.board[i]):
                self.board[i] = np.zeros(BOARD_WIDTH)
                lines_cleared += 1
        self.score += self.get_reward(lines_cleared)
        return lines_cleared

    def get_reward(self, lines_cleared):
        rewards = {0: 0, 1: 40, 2: 100, 3: 300, 4: 1200}
        neatness_score = self.calculate_neatness()
        return rewards.get(lines_cleared, 0) + neatness_score

    def calculate_neatness(self):
        # Reward for neatness: fewer holes and more complete rows
        holes = 0
        for x in range(BOARD_WIDTH):
            found_block = False
            for y in range(BOARD_HEIGHT):
                if self.board[y][x] == 1:
                    found_block = True
                elif found_block:
                    holes += 1
        return -holes  # Negative reward for holes

    def draw_board(self):
        screen.fill(COLORS['BACKGROUND'])
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                color = COLORS['BACKGROUND'] if self.board[y][x] == 0 else COLORS['I']
                pygame.draw.rect(screen, color, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()

# DQN Model
def build_dqn(input_shape, action_size):
    model = Sequential()
    model.add(Dense(32, input_shape=input_shape, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model

# Main game loop
def main():
    clock = pygame.time.Clock()
    tetris = Tetris()
    dqn = build_dqn((BOARD_HEIGHT, BOARD_WIDTH), len(SHAPES))
    replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
    epsilon = 1.0

    for episode in range(NUM_EPISODES):
        tetris = Tetris()
        total_reward = 0

        while not tetris.game_over:
            action = np.random.choice(len(SHAPES)) if np.random.rand() <= epsilon else np.argmax(dqn.predict(tetris.board.reshape(1, BOARD_HEIGHT, BOARD_WIDTH)))

            # Implement action logic here (e.g., move piece, rotate, etc.)
            # Update the board and check for game over condition

            lines_cleared = tetris.clear_lines()
            reward = tetris.get_reward(lines_cleared)
            total_reward += reward

            # Store experience in replay memory
            replay_memory.append((tetris.board, action, reward, tetris.board, tetris.game_over))

            if len(replay_memory) > BATCH_SIZE:
                minibatch = random.sample(replay_memory, BATCH_SIZE)
                for state, action, reward, next_state, done in minibatch:
                    target = reward
                    if not done:
                        target += GAMMA * np.amax(dqn.predict(next_state.reshape(1, BOARD_HEIGHT, BOARD_WIDTH)))
                    target_f = dqn.predict(state.reshape(1, BOARD_HEIGHT, BOARD_WIDTH))
                    target_f[0][action] = target
                    dqn.fit(state.reshape(1, BOARD_HEIGHT, BOARD_WIDTH), target_f, epochs=1, verbose=0)

            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY

            tetris.draw_board()
            clock.tick(FPS)

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    pygame.quit()

if __name__ == "__main__":
    main()
