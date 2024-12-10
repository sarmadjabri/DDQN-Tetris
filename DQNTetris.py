import pygame
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from collections import deque

# Constants
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BLOCK_SIZE = 40  # Increased block size for better visibility
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
        self.piece_pos = [0, BOARD_WIDTH // 2]  # Initial position of the piece (top center)

    def new_piece(self):
        shape = random.choice(list(SHAPES.keys()))
        return SHAPES[shape]

    def rotate(self):
        # Try rotating the piece and check if it fits
        self.current_piece = np.rot90(self.current_piece)
        if not self.valid_move(self.piece_pos[0], self.piece_pos[1], self.current_piece):
            self.current_piece = np.rot90(self.current_piece, 3)  # Rotate back if invalid

    def valid_move(self, row, col, piece):
        # Check if the current piece can be placed at (row, col) position
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x]:
                    if x + col < 0 or x + col >= BOARD_WIDTH or y + row >= BOARD_HEIGHT or self.board[y + row][x + col]:
                        return False
        return True

    def move_left(self):
        if self.valid_move(self.piece_pos[0], self.piece_pos[1] - 1, self.current_piece):
            self.piece_pos[1] -= 1

    def move_right(self):
        if self.valid_move(self.piece_pos[0], self.piece_pos[1] + 1, self.current_piece):
            self.piece_pos[1] += 1

    def move_down(self):
        if self.valid_move(self.piece_pos[0] + 1, self.piece_pos[1], self.current_piece):
            self.piece_pos[0] += 1
            return False  # Can continue falling
        else:
            self.lock_piece()
            return True  # Can't fall anymore

    def lock_piece(self):
        for y in range(len(self.current_piece)):
            for x in range(len(self.current_piece[y])):
                if self.current_piece[y][x]:
                    self.board[self.piece_pos[0] + y][self.piece_pos[1] + x] = 1
        self.clear_lines()

    def clear_lines(self):
        lines_cleared = 0
        for i in range(BOARD_HEIGHT):
            if all(self.board[i]):
                self.board[i] = np.zeros(BOARD_WIDTH)
                lines_cleared += 1
        self.score += self.get_reward(lines_cleared)
        return lines_cleared

    def get_reward(self, lines_cleared):
        # Reward for lines cleared
        rewards = {0: 0, 1: 40, 2: 100, 3: 300, 4: 1200}
        line_clear_reward = rewards.get(lines_cleared, 0)

        # Neatness reward: Penalize holes and messy board
        holes = self.calculate_holes()
        neatness_reward = -holes * 10  # Each hole gives a penalty of 10

        # Stack height reward: Reward for keeping the stack low
        stack_height_reward = -self.calculate_stack_height()  # Negative because we want to minimize stack height

        # Smoothness reward: Penalize rough columns with gaps
        smoothness_reward = -self.calculate_smoothness() * 2  # This penalizes roughness heavily

        # Total reward is the sum of the individual components
        total_reward = line_clear_reward + neatness_reward + stack_height_reward + smoothness_reward
        return total_reward

    def calculate_holes(self):
        holes = 0
        for x in range(BOARD_WIDTH):
            found_block = False
            for y in range(BOARD_HEIGHT):
                if self.board[y][x] == 1:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes

    def calculate_stack_height(self):
        max_height = 0
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                if self.board[y][x] == 1:
                    max_height = max(max_height, BOARD_HEIGHT - y)
        return max_height

    def calculate_smoothness(self):
        smoothness = 0
        for x in range(BOARD_WIDTH - 1):
            for y in range(BOARD_HEIGHT):
                if self.board[y][x] == 1 and self.board[y][x + 1] == 1:
                    smoothness += 1  # This counts adjacent blocks in the same row
        return smoothness

    def draw_board(self):
        screen.fill(COLORS['BACKGROUND'])
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                color = COLORS['BACKGROUND'] if self.board[y][x] == 0 else COLORS['I']  # Add logic for different shapes
                pygame.draw.rect(screen, color, (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()

# DQN Model
def build_dqn(input_shape, action_size):
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model

# Experience replay method
def train_dqn(dqn, replay_memory, batch_size):
    # Sample a batch from the replay memory
    minibatch = random.sample(replay_memory, batch_size)
    states = np.zeros((batch_size, BOARD_HEIGHT, BOARD_WIDTH))
    next_states = np.zeros((batch_size, BOARD_HEIGHT, BOARD_WIDTH))
    actions, rewards, dones = [], [], []

    for i, (state, action, reward, next_state, done) in enumerate(minibatch):
        states[i] = state
        next_states[i] = next_state
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

    states = states.reshape(batch_size, BOARD_HEIGHT, BOARD_WIDTH)
    next_states = next_states.reshape(batch_size, BOARD_HEIGHT, BOARD_WIDTH)

    # Predict Q-values for the current and next states
    target_f = dqn.predict(states)
    target_next = dqn.predict(next_states)

    # Update the Q-values based on the Bellman equation
    for i in range(batch_size):
        target = rewards[i]
        if not dones[i]:
            target += GAMMA * np.amax(target_next[i])

        target_f[i][actions[i]] = target  # Update the Q-value for the chosen action

    # Train the DQN model on the states and targets
    dqn.fit(states, target_f, epochs=1, verbose=0)

# Main game loop
def main():
    clock = pygame.time.Clock()
    tetris = Tetris()
    dqn = build_dqn((BOARD_HEIGHT, BOARD_WIDTH), 6)  # 6 actions now (left, right, rotate, drop)
    replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
    epsilon = 1.0

    for episode in range(NUM_EPISODES):
        tetris = Tetris()
        total_reward = 0

        while not tetris.game_over:
            state = tetris.board.reshape(1, BOARD_HEIGHT, BOARD_WIDTH)
            action = np.random.choice(6) if np.random.rand() <= epsilon else np.argmax(dqn.predict(state))

            # Execute action
            if action == 0:
                tetris.rotate()
            elif action == 1:
                tetris.move_left()
            elif action == 2:
                tetris.move_right()
            elif action == 3:
                tetris.move_down()

            # Reward and training
            lines_cleared = tetris.clear_lines()
            reward = tetris.get_reward(lines_cleared)
            total_reward += reward

            # Store the experience in replay memory
            next_state = tetris.board.reshape(1, BOARD_HEIGHT, BOARD_WIDTH)
            replay_memory.append((state, action, reward, next_state, tetris.game_over))

            # Update Q-values using experience replay
            if len(replay_memory) > BATCH_SIZE:
                train_dqn(dqn, replay_memory, BATCH_SIZE)

            # Update epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY

            # Draw the board
            tetris.draw_board()
            clock.tick(FPS)

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    pygame.quit()

if __name__ == "__main__":
    main()
