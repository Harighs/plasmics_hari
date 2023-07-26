import random
import time
import os

def initialize_grid(width, height):
    grid = [[0] * width for _ in range(height)]
    return grid

def randomize_grid(grid):
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            grid[row][col] = random.randint(0, 1)


def print_grid(grid):
    for row in grid:
        for cell in row:
            print("■" if cell else "□", end="")
            print("", end="")
        print()

def count_neighbors(grid, row, col):
    count = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            if (0 <= row + i < len(grid)) and (0 <= col + j < len(grid[row])):
                count += grid[row + i][col + j]
    return count

def update_grid(grid):
    new_grid = [[0] * len(grid[0]) for _ in range(len(grid))]
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            cell = grid[row][col]
            neighbors = count_neighbors(grid, row, col)
            if cell and (neighbors < 2 or neighbors > 3):
                new_grid[row][col] = 0
            elif not cell and neighbors == 3:
                new_grid[row][col] = 1
            else:
                new_grid[row][col] = cell
    return new_grid

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def main():
    width = 160
    height = 80
    grid = initialize_grid(width, height)
    randomize_grid(grid)

    while True:
        clear_screen()
        print_grid(grid)
        grid = update_grid(grid)
        time.sleep(0.2)

if __name__ == "__main__":
    main()
