import pygame
import random

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
grid = [[random.choice([0, 1]) for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]


def get_neighbor_count(grid, x, y):
    count = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            neighbor_x = (x + j + GRID_WIDTH) % GRID_WIDTH
            neighbor_y = (y + i + GRID_HEIGHT) % GRID_HEIGHT
            count += grid[neighbor_y][neighbor_x]
    count -= grid[y][x]
    return count


def update_grid(grid):
    new_grid = [[0] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            count = get_neighbor_count(grid, x, y)
            if grid[y][x] == 1:
                if count in [2, 3]:
                    new_grid[y][x] = 1
            elif count == 3:
                new_grid[y][x] = 1
    return new_grid


def draw_grid():
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if grid[y][x] == 1:
                pygame.draw.rect(screen, WHITE, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))


# Game loop
running = True
paused = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_c:
                grid = [[0] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
            elif event.key == pygame.K_r:
                grid = [[random.choice([0, 1]) for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
            # mouse key:
        if pygame.mouse.get_pressed()[0]:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            grid[mouse_y // CELL_SIZE][mouse_x // CELL_SIZE] = 1

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