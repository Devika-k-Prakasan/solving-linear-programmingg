import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

num_constraints = int(input("Enter the number of constraints: "))

constraints_list = []
for i in range(num_constraints):
    a = float(input(f"Enter coefficient of x in constraint {i+1}: "))
    b = float(input(f"Enter coefficient of y in constraint {i+1}: "))
    c = float(input(f"Enter RHS value of constraint {i+1} (ax + by ≤ c): "))
    constraints_list.append((a, b, c))

obj_x = float(input("Enter coefficient of x in objective function Z = ax + by: "))
obj_y = float(input("Enter coefficient of y in objective function Z = ax + by: "))
opt_type = input("Enter 'max' for maximization or 'min' for minimization: ").strip().lower()

x = np.linspace(0, 100, 400)

def find_intersections():
    points = []
    
    for (i, (a1, b1, c1)), (j, (a2, b2, c2)) in combinations(enumerate(constraints_list), 2):
        A = np.array([[a1, b1], [a2, b2]])
        B = np.array([c1, c2])
        
        try:
            sol = np.linalg.solve(A, B)
            if sol[0] >= 0 and sol[1] >= 0:
                points.append(tuple(sol))
        except np.linalg.LinAlgError:
            pass

    for a, b, c in constraints_list:
        if a != 0:
            points.append((c / a, 0))
        if b != 0:
            points.append((0, c / b))
    
    points.append((0, 0))

    feasible_points = [p for p in points if all(a * p[0] + b * p[1] <= c + 1e-6 for (a, b, c) in constraints_list)]
    
    return list(set(feasible_points))


def objective_function(x, y):
    return obj_x * x + obj_y * y

points = find_intersections()

if not points:
    print("No feasible region found.")
    exit()

plt.figure(figsize=(8, 6))
for i, (a, b, c) in enumerate(constraints_list):
    if b != 0:
        y = (c - a*x) / b
        plt.plot(x, y, label=f'Constraint {i+1}')
    else:
        plt.axvline(x=c/a, label=f'Constraint {i+1}')

y_upper_bound = np.full_like(x, np.inf)
for a, b, c in constraints_list:
    if b != 0:
        y_constraint = (c - a*x) / b
        y_upper_bound = np.minimum(y_upper_bound, y_constraint)

y_upper_bound[y_upper_bound == np.inf] = 0

plt.fill_between(x, y_upper_bound, 0, color='green', alpha=0.4, label="Feasible Region")

optimal_point = None
optimal_value = np.inf if opt_type == "min" else -np.inf

print("\nFeasible Points and Corresponding Z-values:")
for x_val, y_val in points:
    z = objective_function(x_val, y_val)
    plt.scatter(x_val, y_val, color='red')
    plt.text(x_val, y_val, f'({x_val:.2f}, {y_val:.2f})', fontsize=10, verticalalignment='bottom')
    
    print(f'Point: ({x_val:.2f}, {y_val:.2f}) -> Z = {z:.2f}')

    if (opt_type == "max" and z > optimal_value) or (opt_type == "min" and z < optimal_value):
        optimal_value = z
        optimal_point = (x_val, y_val)

if optimal_point:
    plt.scatter(*optimal_point, color='blue', marker='o', label=f'Optimal: {optimal_point}')
    print(f'\nOptimal Solution: x = {optimal_point[0]:.2f}, y = {optimal_point[1]:.2f}, Z = {optimal_value:.2f}')
else:
    print("\nNo optimal solution found.")

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Graphical Solution of LPP')
plt.xlim((0, max(10, max(x) + 2)))
plt.ylim((0, max(10, max(y_upper_bound) + 2)))
plt.grid()
plt.show()
