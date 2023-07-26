import pygame
import numpy as np

# Initialize the Pygame library
pygame.init()

# Set the width and height of the screen (in pixels)
WIDTH = 800
HEIGHT = 600

# Set the number of cells in each direction
CELL_SIZE = 5
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE

# Set the colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Game of Life for PLASMICS")

# Create the grid
grid = np.random.choice([0, 1], size=(GRID_HEIGHT, GRID_WIDTH))


def get_neighbor_count(grid, x, y):
    return np.sum(grid[(y-1):(y+2), (x-1):(x+2)]) - grid[y, x]

def update_grid(grid):
    counts = np.zeros_like(grid)
    for i in range(-1, 2):
        for j in range(-1, 2):
            neighbor_indices = (np.array([i, j]) + np.indices((GRID_HEIGHT, GRID_WIDTH))) % np.array([GRID_HEIGHT, GRID_WIDTH])
            counts += grid[neighbor_indices[0], neighbor_indices[1]]
    counts -= grid
    new_grid = np.zeros_like(grid)
    new_grid[(grid == 1) & ((counts == 2) | (counts == 3))] = 1
    new_grid[(grid == 0) & (counts == 3)] = 1
    return new_grid


def draw_grid():
    live_cells = np.where(grid == 1)
    for y, x in zip(live_cells[0], live_cells[1]):
        pygame.draw.rect(screen, WHITE, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))


# Game loop
running = True
paused = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                grid = np.zeros_like(grid)
            elif event.key == pygame.K_r:
                grid = np.random.choice([0, 1], size=(GRID_HEIGHT, GRID_WIDTH))

    if not paused:
        grid = update_grid(grid)

    # Fill the screen with black color
    screen.fill(BLACK)

    # Draw the grid
    draw_grid()

    # Update the display
    pygame.display.update()

# Quit the game
pygame.quit()
