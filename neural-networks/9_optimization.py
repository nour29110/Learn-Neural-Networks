# Optimization and Derivatives

import matplotlib.pyplot as plt
import numpy as np

# Defining a simple quadratic function
def f(x):
    return 2 * x**2  # The function f(x) = 2x²

# Generating x values from 0 to 50 with a small step for smooth plotting
x = np.arange(0, 50, 0.001)
y = f(x)  # Calculate y values using the defined function

# Plot the function f(x)
plt.plot(x, y)

# Define colors for different tangent lines
colors = ['k', 'g', 'r', 'b', 'c']

# Function to calculate the equation of a tangent line at a given point
def approximate_tangent_line(x, approximate_derivative, b):
    return approximate_derivative * x + b  # y = mx + b form

# Looping to calculate tangent lines at 5 different points on the curve
for i in range(5):
    p2_delta = 0.0001  # A small value to approximate the derivative
    x1 = i  # Choose the point where we will calculate the derivative
    x2 = x1 + p2_delta  # A nearby point for derivative approximation

    y1 = f(x1)  # Function value at x1
    y2 = f(x2)  # Function value at x2

    # Print the coordinates of the points used for derivative calculation
    print((x1, y1), (x2, y2))

    # Calculate the approximate derivative using the difference quotient
    approximate_derivative = (y2 - y1) / (x2 - x1)
    # Calculate the y-intercept of the tangent line using point-slope form
    b = y2 - approximate_derivative * x2

    # Define points around x1 to plot the tangent line
    to_plot = [x1 - 0.9, x1, x1 + 0.9]

    # Scatter the point on the plot where the tangent is calculated
    plt.scatter(x1, y1, c=colors[i])
    # Plot the tangent line using the approximate derivative and intercept
    plt.plot(to_plot,
             [approximate_tangent_line(point, approximate_derivative, b)
              for point in to_plot],
             c=colors[i])

# Print the approximate derivative at the last calculated point
print('Approximate derivative for f(x)',
      f'where x = {x1} is {approximate_derivative}')

# Show the plot with the function and tangent lines
plt.show()