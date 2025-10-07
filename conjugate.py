import numpy as np


def rosenbrock(x, y):
    return (1 - x)**2 + 100*(y - x**2)**2


def rosenbrock_grad(x, y):
    df_dx = -2*(1-x) - 400*x*(y-x**2)
    df_dy = 200 * (y - x**2)
    return np.array([df_dx, df_dy])


def conjugate_gradient_descent(f, grad, x0, lr=1e-3, tol=1e-6, max_iter=10000):
    x = np.array(x0, dtype=float)
    g = grad(*x)
    d = -g
    history = [x.copy()]
    for i in range(max_iter):
        # line search
        alpha = lr
        while f(*(x + alpha*d)) > f(*x) - 0.5 * alpha * np.dot(g, g):
            alpha *= 0.5

        # update x
        x_new = x + alpha * d
        g_new = grad(*x_new)

        # Iterations every 50
        if i % 50 == 0:
            print(
                f"Iter: {i:6d}, X: {x_new[0]:10.6f}, y: {x_new[1]:10.6f}, f(x, y): {f(*x_new):12.6e}")

        # convergence check
        if np.linalg.norm(g_new) < tol:
            print(f"Converged in {i+1} iterations.")
            break

        # Compute beta (FR formula)
        beta = np.dot(g_new, g_new) / np.dot(g, g)

        # Update direction
        d = -g_new + beta * d

        # Prepare for next iteration
        x, g = x_new, g_new
        history.append(x.copy())

    return np.array(history), x


start = np.random.uniform(-2, 2, 2)  # random numbers to start
path, x_min = conjugate_gradient_descent(rosenbrock, rosenbrock_grad, start)

print("Start:", start)
print("Minimum found at:", x_min)
print("Function value:", rosenbrock(*x_min))
