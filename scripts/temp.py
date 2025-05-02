import pygame
import numpy as np
import sys
import subprocess
import os

# Initialize pygame
pygame.init()

# Set the canvas dimensions and cell size
CANVAS_SIZE = 280
GRID_SIZE = 28
CELL_SIZE = CANVAS_SIZE // GRID_SIZE

# Set up colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)

# Set up the display
screen = pygame.display.set_mode((CANVAS_SIZE, CANVAS_SIZE + 100))
pygame.display.set_caption("Digit Drawing for Recognition")

# Create a grid to store the drawn digit (28x28)
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

# Create a font for text
font = pygame.font.SysFont('Arial', 24)

def draw_grid():
    # Fill the screen with white
    screen.fill(BLACK)
    
    # Draw the grid lines
    for i in range(GRID_SIZE + 1):
        pygame.draw.line(screen, GRAY, (0, i * CELL_SIZE), (CANVAS_SIZE, i * CELL_SIZE))
        pygame.draw.line(screen, GRAY, (i * CELL_SIZE, 0), (i * CELL_SIZE, CANVAS_SIZE))
    
    # Draw the filled cells
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if grid[y][x] > 0:
                pygame.draw.rect(screen, 
                                (int(grid[y][x] * 255), int(grid[y][x] * 255), int(grid[y][x] * 255)),
                                (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    # Draw the buttons
    pygame.draw.rect(screen, (100, 200, 100), (20, CANVAS_SIZE + 20, 120, 60))
    pygame.draw.rect(screen, (200, 100, 100), (150, CANVAS_SIZE + 20, 120, 60))
    
    recognize_text = font.render('Recognize', True, BLACK)
    clear_text = font.render('Clear', True, BLACK)
    
    screen.blit(recognize_text, (30, CANVAS_SIZE + 40))
    screen.blit(clear_text, (180, CANVAS_SIZE + 40))

def send_to_cpp(grid_data):
    """Flatten and send the grid data to the C++ program via stdin"""
    flattened_data = grid_data.flatten().tolist()
    
    # Format the data as a single line of space-separated values
    data_str = ' '.join(map(str, flattened_data))
    
    try:
        # Start the C++ process
        cpp_process = subprocess.Popen(
            [r"C:\CLionProjects\nn\neural_network\cmake-build-debug\predictor.exe"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send the data and get the result
        stdout, stderr = cpp_process.communicate(input=data_str)
        
        if stderr:
            print(f"Error from C++ program: {stderr}")
            return "Error"
        
        # Return the prediction
        return stdout.strip()
    
    except Exception as e:
        print(f"Error communicating with C++ program: {e}")
        return "Error"

def main():
    running = True
    drawing = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    x, y = event.pos
                    
                    # Check if the click is within the canvas area
                    if 0 <= y < CANVAS_SIZE and 0 <= x < CANVAS_SIZE:
                        drawing = True
                    
                    # Check if the Recognize button was clicked
                    elif CANVAS_SIZE + 20 <= y <= CANVAS_SIZE + 80:
                        if 20 <= x <= 140:  # Recognize button
                            prediction = send_to_cpp(grid)
                            print(f"Predicted digit: {prediction}")
                        elif 150 <= x <= 270:  # Clear button
                            grid.fill(0)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    drawing = False
            
            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    x, y = event.pos
                    if 0 <= y < CANVAS_SIZE and 0 <= x < CANVAS_SIZE:
                        # Calculate the grid cell
                        grid_x = x // CELL_SIZE
                        grid_y = y // CELL_SIZE
                        
                        # Set the cell value (and surrounding cells with less intensity)
                        radius = 1
                        for dy in range(-radius, radius + 1):
                            for dx in range(-radius, radius + 1):
                                nx, ny = grid_x + dx, grid_y + dy
                                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                                    # Calculate distance from center
                                    distance = np.sqrt(dx**2 + dy**2)
                                    intensity = max(0, 1.0 - distance * 0.5)
                                    grid[ny][nx] = min(1.0, grid[ny][nx] + intensity)
        
        draw_grid()
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    main()