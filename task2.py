import numpy as np
import math


def rosenbrock(x, y):
    return (1 - x)**2 + 100*(y - x**2)**2


def rosenbrock_grad(x, y):
    df_dx = -2*(1-x) - 400*x*(y-x**2)
    df_dy = 200 * (y - x**2)
    return np.array([df_dx, df_dy])


def gradient_descent(lr=0.001, epochs=10000, tolerance=1e-6):
    x, y = np.random.uniform(-2, 2, 2)
    print(f"Starting from random point x = {x:.4f}, y = {y:.4f}")
    for i in range(epochs):
        grad = rosenbrock_grad(x, y)
        new_x = x - lr*grad[0]
        new_y = y - lr * grad[1]

        if np.linalg.norm([new_x - x, new_y - y]) < tolerance:
            print(f"Converged after {i} iterations")
            break

        x, y = new_x, new_y

        # Print progress every 1000 steps
        if i % 1000 == 0:
            print(
                f"Iter {i}: x={x:.5f}, y={y:.5f}, f(x,y)={rosenbrock(x,y):.8f}")

    return x, y


min_x, min_y = gradient_descent(lr=0.001, epochs=50000)
print(f"\nMinimum found at: x={min_x:.6f}, y={min_y:.6f}")
print(f"Function value: f(x,y)={rosenbrock(min_x, min_y):.8f}")
