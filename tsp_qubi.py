import numpy as np
import dimod
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

 # %% 
# User input for number of cities
num_cities = int(input("Enter the number of cities: "))
num_steps = num_cities  # One step per city

# User input for distance matrix (Each row should be entered separately)
print("Enter the distance matrix row by row. Each row should be space-separated.")

# Initialize the distance matrix
distance_matrix = np.zeros((num_cities, num_cities))

# Take matrix input row by row
for i in range(num_cities):
    while True:
        try:
            row = input(f"Enter row {i + 1} (space-separated distances): ")
            row_values = list(map(float, row.split()))  # Convert to a list of floats
            if len(row_values) != num_cities:
                print(f"Error: There should be {num_cities} values in row {i + 1}. Please try again.")
                continue
            distance_matrix[i] = row_values
            break
        except ValueError:
            print("Invalid input. Please enter numeric values separated by spaces.")

# Ensure the matrix is symmetric (distance[i][j] == distance[j][i]) and no self-loops
np.fill_diagonal(distance_matrix, 0)

# Plot Heatmap of Distance Matrix with correct formatting
plt.figure(figsize=(8, 6))
sns.heatmap(distance_matrix, annot=True, cmap="YlGnBu", fmt=".2f", cbar=True)
plt.title("Distance Matrix Heatmap")
plt.xlabel("Cities")
plt.ylabel("Cities")
plt.tight_layout()
plt.show()

# Define QUBO dictionary
Q = {}

# Large penalty coefficient
A = 1000  

# Constraint 1: Each city is visited exactly once
for i in range(num_cities):
    for t in range(num_steps):
        Q[(i * num_steps + t, i * num_steps + t)] = -A  # Favor selecting one per row
    for t1 in range(num_steps):
        for t2 in range(t1 + 1, num_steps):
            Q[(i * num_steps + t1, i * num_steps + t2)] = A  # Penalize multiple visits

# Constraint 2: Each time slot has exactly one city
for t in range(num_steps):
    for i in range(num_cities):
        Q[(i * num_steps + t, i * num_steps + t)] = -A  # Favor selecting one per column
    for i1 in range(num_cities):
        for i2 in range(i1 + 1, num_cities):
            Q[(i1 * num_steps + t, i2 * num_steps + t)] = A  # Penalize multiple selections

# Objective function: Minimize travel cost
for i in range(num_cities):
    for j in range(num_cities):
        if i != j:
            for t in range(num_steps - 1):
                Q[(i * num_steps + t, j * num_steps + t + 1)] = distance_matrix[i, j]

# Solve using Simulated Annealing
sampler = dimod.SimulatedAnnealingSampler()
response = sampler.sample_qubo(Q, num_reads=100)

# Get the best solution
best_sample = response.first.sample
print("Best route:")
route = [None] * num_steps
for t in range(num_steps):
    for i in range(num_cities):
        if best_sample[i * num_steps + t] == 1:
            route[t] = i
            print(f"Step {t+1}: Visit City {i}")

# Print energy (objective value)
print("Energy (Cost):", response.first.energy)

# 3D visualization of the cities and optimal route
city_coordinates = np.random.rand(num_cities, 3) * 100  # Random 3D coordinates for cities

# Plot the route on a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot cities
ax.scatter(city_coordinates[:, 0], city_coordinates[:, 1], city_coordinates[:, 2], color='red', s=100, label="Cities")

# Plot the route
for t in range(1, num_steps):
    city1 = route[t - 1]
    city2 = route[t]
    ax.plot([city_coordinates[city1, 0], city_coordinates[city2, 0]], 
            [city_coordinates[city1, 1], city_coordinates[city2, 1]], 
            [city_coordinates[city1, 2], city_coordinates[city2, 2]], color='blue')

ax.set_title("Optimal Route (3D Visualization)")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z Coordinate")
ax.legend()
plt.tight_layout()
plt.show()
 # %% 
# Save the results and plots in a PDF
with PdfPages("TSP_Results_3D.pdf") as pdf:
    # Plot Heatmap of Distance Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(distance_matrix, annot=True, cmap="YlGnBu", fmt=".2f", cbar=True)
    plt.title("Distance Matrix Heatmap")
    plt.xlabel("Cities")
    plt.ylabel("Cities")
    plt.tight_layout()
    pdf.savefig()
    plt.close()
 # %% 

    # Plot 3D Route
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(city_coordinates[:, 0], city_coordinates[:, 1], city_coordinates[:, 2], color='red', s=100, label="Cities")
    for t in range(1, num_steps):
        city1 = route[t - 1]
        city2 = route[t]
        ax.plot([city_coordinates[city1, 0], city_coordinates[city2, 0]], 
                [city_coordinates[city1, 1], city_coordinates[city2, 1]], 
                [city_coordinates[city1, 2], city_coordinates[city2, 2]], color='blue')

    ax.set_title("Optimal Route (3D Visualization)")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Save results summary
    plt.figure(figsize=(8, 6))
    plt.text(0.1, 0.8, f"Best Route: {route}", fontsize=12)
    plt.text(0.1, 0.7, f"Energy (Cost): {response.first.energy}", fontsize=12)
    plt.axis('off')
    pdf.savefig()
    plt.close()