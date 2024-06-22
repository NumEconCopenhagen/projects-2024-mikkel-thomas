# Write your code here
import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})

def objective_function(x1,x2,y1,y2):
    return np.sqrt((x1 - y1)**2 + (x2 - y2)**2)

def compute_coefficients(X, y1, y2, coeff_type):
    """Compute the coefficients for barycentric coordinates for the given type of coefficient
    Args:
        X (np.ndarray): The array of points
        y1 (float): The x-coordinate of the point to compute the coefficients for
        y2 (float): The y-coordinate of the point to compute the coefficients for
        coeff_type (str): The type of coefficient to compute

    Returns:
        np.ndarray: List of coefficients for the given type of coefficient

    """
    coeff = np.inf
    arg_min = None
    for x1, x2 in X:
        if coeff_type == 'A':
            cond = x1 > y1 and x2 > y2
        elif coeff_type == 'B':
            cond = x1 > y1 and x2 < y2
        elif coeff_type == 'C':
            cond = x1 < y1 and x2 < y2
        elif coeff_type == 'D':
            cond = x1 < y1 and x2 > y2
        if cond:
            coeff_temp = objective_function(x1,x2,y1,y2)
            if coeff_temp < coeff:
                coeff = coeff_temp
                arg_min = (x1,x2)
    coeff = np.nan if coeff == np.inf else coeff
    arg_min = (np.nan, np.nan) if arg_min is None else arg_min
    return arg_min, coeff

def plot_fig(coeff_cords, A, B, C, D, X, y):
    fig = plt.figure(figsize=(6,6))

    ax = fig.add_subplot(1,1,1)

    # Extract x and y coordinates
    x_coords = [coord[0] for coord in coeff_cords]
    y_coords = [coord[1] for coord in coeff_cords]

    ax.plot(X[:,0], X[:,1], 'o', label='X', zorder = 1)
    ax.scatter(x_coords, y_coords, color='red', label='Coefficients', zorder = 2)
    ax.scatter(y[0], y[1], color='orange', label='y', zorder = 2)

    # Plotting ABC triangle
    ax.plot([A[0], B[0]], [A[1], B[1]], 'k-', zorder = 1)  # Line from A to B
    ax.plot([B[0], C[0]], [B[1], C[1]], 'k-', zorder = 1)  # Line from B to C
    ax.plot([C[0], A[0]], [C[1], A[1]], 'k-', zorder = 1, label = 'ABC')  # Line from C to A

    # Plotting CDA triangle
    ax.plot([C[0], D[0]], [C[1], D[1]], 'g-', zorder = 1)  # Line from C to D
    ax.plot([D[0], A[0]], [D[1], A[1]], 'g-', zorder = 1)  # Line from D to A
    ax.plot([C[0], A[0]], [C[1], A[1]], 'g-', zorder = 1, label = 'CDA')  # Line from C to A

    # Annotating the vertices
    ax.text(A[0], A[1], 'A', fontsize=12, ha='right')
    ax.text(B[0], B[1], 'B', fontsize=12, ha='right')
    ax.text(C[0], C[1], 'C', fontsize=12, ha='right')
    ax.text(D[0], D[1], 'D', fontsize=12, ha='right')
    ax.legend(fontsize=12)

    plt.show()

def barycentric_cord(A,B,C,y):
    """Question 2: Computes the  barycentric coordinates of a point y with respect to triangle ABC."""
    r1 = ((B[1] - C[1])*(y[0] - C[0]) + (C[0] - B[0])*(y[1] - C[1])) / ((B[1] - C[1])*(A[0] - C[0]) + (C[0] - B[0])*(A[1] - C[1]))
    r2 = ((C[1] - A[1])*(y[0] - C[0]) + (A[0] - C[0])*(y[1] - C[1])) / ((B[1] - C[1])*(A[0] - C[0]) + (C[0] - B[0])*(A[1] - C[1]))
    r3 = 1 - r1 - r2

    is_inside = (0 <= r1 <= 1) and (0 <= r2 <= 1) and (0 <= r3 <= 1)
    return r1, r2, r3, is_inside


def barycentric_approx(X, y, f):
    """Question 4: Full implementation of the barycentric approximation."""
    coeff_list = ['A', 'B', 'C', 'D']
    coeff_cords = []

    for i, coeff_type in enumerate(coeff_list):
        arg_min, _ = compute_coefficients(X, y[0], y[1], coeff_type)
        coeff_cords.append(arg_min)

    A = coeff_cords[0]
    B = coeff_cords[1]
    C = coeff_cords[2]
    D = coeff_cords[3]
    #print(coeff_cords)
    r1_ABC, r2_ABC, r3_ABC, is_inside_ABC = barycentric_cord(A,B,C,y)
    r1_CDA, r2_CDA, r3_CDA, is_inside_CDA = barycentric_cord(C,D,A,y)

    if is_inside_ABC:
        approx_func = r1_ABC*f(y) + r2_ABC*f(y) + r3_ABC*f(y)
    elif is_inside_CDA:
        approx_func = r1_CDA*f(y) + r2_CDA*f(y) + r3_CDA*f(y)
    else:
        approx_func = np.nan

    print(f"Approimation of f(y) at y = {y}")
    print(f"Approximation of f(y) = {approx_func}")
    print(f"Actual value of f(y) = {f(y)}")
    print(f"Approximation error = {np.abs(approx_func - f(y))}")
    # return approx_func