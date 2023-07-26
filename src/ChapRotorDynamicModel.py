import numpy as np
from scipy.integrate import cumtrapz, trapz, solve_ivp
import pickle
from tqdm import tqdm
from numba import njit

dt = 0.01
q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

b = 1
c = 0.5
l = 0.5
m1 = 1.0
m2 = 10.0
I1 = m1 * 2 * b ** 2 / 3
I3 = m2 * 4 * l ** 2 / 3

t_max = 300.0
t_sim = np.linspace(0, t_max, int(t_max / dt) + 1)

status_bar = None
status = 0.0

filename = 'chap_rotor_dynamic_model.pkl'


def update_params(params):
    global b, c, l, I1, I3, m1, m2
    b, c, l, I1, I3, m1, m2 = params


@njit
def dynamic_equations(t, q):
    u1, u2, u3, th, phi, x, y = q

    u1_dot = (- b ** 4 * l ** 3 * m2 ** 3 * u3 ** 2 * np.cos(phi) + I3 * b ** 4 * l * m1 * m2 * u3 ** 2 * np.cos(
        phi) + I3 * b ** 4 * l * m2 ** 2 * u3 ** 2 * np.cos(phi) + 2 * b ** 3 * c * l ** 3 * m2 ** 3 * u3 ** 2 * np.cos(
        phi) - 2 * I3 * b ** 3 * c * l * m2 ** 2 * u3 ** 2 * np.cos(
        phi) - 2 * b ** 3 * l ** 3 * m2 ** 3 * u2 * u3 * np.cos(
        phi) - b ** 3 * l ** 2 * m1 * m2 ** 2 * u2 ** 2 - b ** 3 * l ** 2 * m2 ** 3 * u2 ** 2 + 2 * I3 * b ** 3 * l * m1 * m2 * u2 * u3 * np.cos(
        phi) + 2 * I3 * b ** 3 * l * m2 ** 2 * u2 * u3 * np.cos(
        phi) + I3 * b ** 3 * m1 ** 2 * u2 ** 2 + 2 * I3 * b ** 3 * m1 * m2 * u2 ** 2 + I3 * b ** 3 * m2 ** 2 * u2 ** 2 - b ** 2 * c ** 2 * l ** 3 * m2 ** 3 * u3 ** 2 * np.cos(
        phi) + I3 * b ** 2 * c ** 2 * l * m2 ** 2 * u3 ** 2 * np.cos(
        phi) + 4 * b ** 2 * c * l ** 3 * m2 ** 3 * u2 * u3 * np.cos(
        phi) + b ** 2 * c * l ** 2 * m1 * m2 ** 2 * u2 ** 2 * np.cos(
        phi) ** 2 + b ** 2 * c * l ** 2 * m1 * m2 ** 2 * u2 ** 2 - (u1 * np.sin(
        2 * phi) * b ** 2 * c * l ** 2 * m1 * m2 ** 2 * u2) / 2 + 3 * b ** 2 * c * l ** 2 * m2 ** 3 * u2 ** 2 - 4 * I3 * b ** 2 * c * l * m2 ** 2 * u2 * u3 * np.cos(
        phi) - 3 * I3 * b ** 2 * c * m1 * m2 * u2 ** 2 - 3 * I3 * b ** 2 * c * m2 ** 2 * u2 ** 2 - b ** 2 * l ** 3 * m2 ** 3 * u2 ** 2 * np.cos(
        phi) + I3 * b ** 2 * l * m1 * m2 * u2 ** 2 * np.cos(phi) + I3 * b ** 2 * l * m2 ** 2 * u2 ** 2 * np.cos(
        phi) + I1 * I3 * b ** 2 * l * m2 * u3 ** 2 * np.cos(phi) - 2 * b * c ** 2 * l ** 3 * m2 ** 3 * u2 * u3 * np.cos(
        phi) - b * c ** 2 * l ** 2 * m1 * m2 ** 2 * u2 ** 2 * np.cos(
        phi) ** 2 - 3 * b * c ** 2 * l ** 2 * m2 ** 3 * u2 ** 2 + 2 * I3 * b * c ** 2 * l * m2 ** 2 * u2 * u3 * np.cos(
        phi) + I3 * b * c ** 2 * m1 * m2 * u2 ** 2 + 3 * I3 * b * c ** 2 * m2 ** 2 * u2 ** 2 + 2 * b * c * l ** 3 * m2 ** 3 * u2 ** 2 * np.cos(
        phi) - 2 * I3 * b * c * l * m2 ** 2 * u2 ** 2 * np.cos(phi) + I1 * b * l ** 2 * m2 ** 2 * u2 ** 2 * np.cos(
        phi) ** 2 - I1 * b * l ** 2 * m2 ** 2 * u2 ** 2 - (I1 * u1 * np.sin(
        2 * phi) * b * l ** 2 * m2 ** 2 * u2) / 2 + 2 * I1 * I3 * b * l * m2 * u2 * u3 * np.cos(
        phi) + I1 * I3 * b * m1 * u2 ** 2 + I1 * I3 * b * m2 * u2 ** 2 + c ** 3 * l ** 2 * m2 ** 3 * u2 ** 2 - I3 * c ** 3 * m2 ** 2 * u2 ** 2 - c ** 2 * l ** 3 * m2 ** 3 * u2 ** 2 * np.cos(
        phi) + I3 * c ** 2 * l * m2 ** 2 * u2 ** 2 * np.cos(phi) - I1 * c * l ** 2 * m2 ** 2 * u2 ** 2 * np.cos(
        phi) ** 2 + I1 * c * l ** 2 * m2 ** 2 * u2 ** 2 - I1 * I3 * c * m2 * u2 ** 2 + I1 * I3 * l * m2 * u2 ** 2 * np.cos(
        phi)) / (b ** 2 * (
            - b ** 2 * l ** 2 * m1 * m2 ** 2 - b ** 2 * l ** 2 * m2 ** 3 + I3 * b ** 2 * m1 ** 2 + 2 * I3 * b ** 2 * m1 * m2 + I3 * b ** 2 * m2 ** 2 + 2 * b * c * l ** 2 * m1 * m2 ** 2 * np.cos(
        phi) ** 2 + 2 * b * c * l ** 2 * m2 ** 3 - 2 * I3 * b * c * m1 * m2 - 2 * I3 * b * c * m2 ** 2 - c ** 2 * l ** 2 * m1 * m2 ** 2 * np.cos(
        phi) ** 2 - c ** 2 * l ** 2 * m2 ** 3 + I3 * c ** 2 * m1 * m2 + I3 * c ** 2 * m2 ** 2 + I1 * l ** 2 * m2 ** 2 * np.cos(
        phi) ** 2 - I1 * l ** 2 * m2 ** 2 + I1 * I3 * m1 + I1 * I3 * m2))

    u2_dot = -(np.sin(phi) * b ** 3 * l ** 3 * m2 ** 3 * u3 ** 2 - I3 * np.sin(
        phi) * b ** 3 * l * m1 * m2 * u3 ** 2 - I3 * np.sin(
        phi) * b ** 3 * l * m2 ** 2 * u3 ** 2 - np.sin(phi) * b ** 2 * c * l ** 3 * m2 ** 3 * u3 ** 2 + I3 * np.sin(
        phi) * b ** 2 * c * l * m1 * m2 * u3 ** 2 + I3 * np.sin(phi) * b ** 2 * c * l * m2 ** 2 * u3 ** 2 + 2 * np.sin(
        phi) * b ** 2 * l ** 3 * m2 ** 3 * u2 * u3 - u1 * b ** 2 * l ** 2 * m1 * m2 ** 2 * u2 - u1 * b ** 2 * l ** 2 * m2 ** 3 * u2 - 2 * I3 * np.sin(
        phi) * b ** 2 * l * m1 * m2 * u2 * u3 - 2 * I3 * np.sin(
        phi) * b ** 2 * l * m2 ** 2 * u2 * u3 + I3 * u1 * b ** 2 * m1 ** 2 * u2 + 2 * I3 * u1 * b ** 2 * m1 * m2 * u2 + I3 * u1 * b ** 2 * m2 ** 2 * u2 - 2 * np.sin(
        phi) * b * c * l ** 3 * m2 ** 3 * u2 * u3 + (np.sin(
        2 * phi) * b * c * l ** 2 * m1 * m2 ** 2 * u2 ** 2) / 2 + u1 * b * c * l ** 2 * m1 * m2 ** 2 * u2 * np.cos(
        phi) ** 2 + u1 * b * c * l ** 2 * m2 ** 3 * u2 + 2 * I3 * np.sin(
        phi) * b * c * l * m1 * m2 * u2 * u3 + 2 * I3 * np.sin(
        phi) * b * c * l * m2 ** 2 * u2 * u3 - I3 * u1 * b * c * m1 * m2 * u2 - I3 * u1 * b * c * m2 ** 2 * u2 + np.sin(
        phi) * b * l ** 3 * m2 ** 3 * u2 ** 2 - I3 * np.sin(phi) * b * l * m1 * m2 * u2 ** 2 - I3 * np.sin(
        phi) * b * l * m2 ** 2 * u2 ** 2 - (np.sin(2 * phi) * c ** 2 * l ** 2 * m1 * m2 ** 2 * u2 ** 2) / 2 - np.sin(
        phi) * c * l ** 3 * m2 ** 3 * u2 ** 2 + I3 * np.sin(phi) * c * l * m1 * m2 * u2 ** 2 + I3 * np.sin(
        phi) * c * l * m2 ** 2 * u2 ** 2) / (b * (
            - b ** 2 * l ** 2 * m1 * m2 ** 2 - b ** 2 * l ** 2 * m2 ** 3 + I3 * b ** 2 * m1 ** 2 + 2 * I3 * b ** 2 * m1 * m2 + I3 * b ** 2 * m2 ** 2 + 2 * b * c * l ** 2 * m1 * m2 ** 2 * np.cos(
        phi) ** 2 + 2 * b * c * l ** 2 * m2 ** 3 - 2 * I3 * b * c * m1 * m2 - 2 * I3 * b * c * m2 ** 2 - c ** 2 * l ** 2 * m1 * m2 ** 2 * np.cos(
        phi) ** 2 - c ** 2 * l ** 2 * m2 ** 3 + I3 * c ** 2 * m1 * m2 + I3 * c ** 2 * m2 ** 2 + I1 * l ** 2 * m2 ** 2 * np.cos(
        phi) ** 2 - I1 * l ** 2 * m2 ** 2 + I1 * I3 * m1 + I1 * I3 * m2))

    u3_dot = (b * l ** 3 * m2 ** 3 * u2 ** 2 * np.sin(phi) - c * l ** 3 * m2 ** 3 * u2 ** 2 * np.sin(
        phi) + I3 * b ** 2 * m1 ** 2 * u1 * u2 + I3 * b ** 2 * m2 ** 2 * u1 * u2 + (
                      I1 * l ** 2 * m2 ** 2 * u2 ** 2 * np.sin(
                  2 * phi)) / 2 + b ** 3 * l ** 3 * m2 ** 3 * u3 ** 2 * np.sin(
        phi) - b ** 2 * l ** 2 * m2 ** 3 * u1 * u2 - b ** 2 * l ** 2 * m1 * m2 ** 2 * u1 * u2 + (
                      I1 * b ** 2 * l ** 2 * m2 ** 2 * u3 ** 2 * np.sin(
                  2 * phi)) / 2 - I3 * b * c * m2 ** 2 * u1 * u2 + 2 * I3 * b ** 2 * m1 * m2 * u1 * u2 - c ** 2 * l ** 2 * m1 * m2 ** 2 * u2 ** 2 * np.sin(
        2 * phi) - I3 * b ** 3 * l * m2 ** 2 * u3 ** 2 * np.sin(phi) + c ** 3 * l * m1 * m2 ** 2 * u2 ** 2 * np.sin(
        phi) + 2 * b ** 2 * l ** 3 * m2 ** 3 * u2 * u3 * np.sin(
        phi) + b * c * l ** 2 * m2 ** 3 * u1 * u2 - b ** 2 * c * l ** 3 * m2 ** 3 * u3 ** 2 * np.sin(
        phi) - I3 * b * l * m2 ** 2 * u2 ** 2 * np.sin(phi) + I3 * c * l * m2 ** 2 * u2 ** 2 * np.sin(
        phi) - I1 * b * l * m2 ** 2 * u1 * u2 * np.cos(phi) - I3 * b * l * m1 * m2 * u2 ** 2 * np.sin(
        phi) + I1 * c * l * m1 * m2 * u2 ** 2 * np.sin(phi) + I3 * c * l * m1 * m2 * u2 ** 2 * np.sin(phi) + (
                      b * c * l ** 2 * m1 * m2 ** 2 * u1 * u2) / 2 - I3 * b * c * m1 * m2 * u1 * u2 + (
                      3 * b * c * l ** 2 * m1 * m2 ** 2 * u2 ** 2 * np.sin(
                  2 * phi)) / 2 - I3 * b ** 3 * l * m1 * m2 * u3 ** 2 * np.sin(
        phi) - 2 * I3 * b ** 2 * l * m2 ** 2 * u2 * u3 * np.sin(phi) - 2 * b * c * l ** 3 * m2 ** 3 * u2 * u3 * np.sin(
        phi) + b ** 3 * c * l ** 2 * m1 * m2 ** 2 * u3 ** 2 * np.sin(
        2 * phi) + I3 * b ** 2 * c * l * m2 ** 2 * u3 ** 2 * np.sin(
        phi) + I1 * b * l ** 2 * m2 ** 2 * u2 * u3 * np.sin(
        2 * phi) - 2 * b * c ** 2 * l * m1 * m2 ** 2 * u2 ** 2 * np.sin(
        phi) + b ** 2 * c * l * m1 * m2 ** 2 * u2 ** 2 * np.sin(phi) + b ** 2 * c * l * m1 ** 2 * m2 * u2 ** 2 * np.sin(
        phi) - (b ** 2 * c ** 2 * l ** 2 * m1 * m2 ** 2 * u3 ** 2 * np.sin(
        2 * phi)) / 2 - I1 * b * l * m1 * m2 * u1 * u2 * np.cos(phi) + (
                      b * c * l ** 2 * m1 * m2 ** 2 * u1 * u2 * np.cos(
                  2 * phi)) / 2 + 2 * I3 * b * c * l * m2 ** 2 * u2 * u3 * np.sin(
        phi) - 2 * I3 * b ** 2 * l * m1 * m2 * u2 * u3 * np.sin(
        phi) - b * c ** 2 * l ** 2 * m1 * m2 ** 2 * u2 * u3 * np.sin(
        2 * phi) + 2 * b ** 2 * c * l ** 2 * m1 * m2 ** 2 * u2 * u3 * np.sin(
        2 * phi) + I3 * b ** 2 * c * l * m1 * m2 * u3 ** 2 * np.sin(
        phi) - b ** 2 * c * l * m1 * m2 ** 2 * u1 * u2 * np.cos(
        phi) - b ** 2 * c * l * m1 ** 2 * m2 * u1 * u2 * np.cos(phi) + 2 * I3 * b * c * l * m1 * m2 * u2 * u3 * np.sin(
        phi)) / (b ** 2 * (
            - b ** 2 * l ** 2 * m1 * m2 ** 2 - b ** 2 * l ** 2 * m2 ** 3 + I3 * b ** 2 * m1 ** 2 + 2 * I3 * b ** 2 * m1 * m2 + I3 * b ** 2 * m2 ** 2 + 2 * b * c * l ** 2 * m1 * m2 ** 2 * np.cos(
        phi) ** 2 + 2 * b * c * l ** 2 * m2 ** 3 - 2 * I3 * b * c * m1 * m2 - 2 * I3 * b * c * m2 ** 2 - c ** 2 * l ** 2 * m1 * m2 ** 2 * np.cos(
        phi) ** 2 - c ** 2 * l ** 2 * m2 ** 3 + I3 * c ** 2 * m1 * m2 + I3 * c ** 2 * m2 ** 2 + I1 * l ** 2 * m2 ** 2 * np.cos(
        phi) ** 2 - I1 * l ** 2 * m2 ** 2 + I1 * I3 * m1 + I1 * I3 * m2))

    th_dot = u2 / b
    phi_dot = u3
    x_dot = u1 * np.cos(th) - u2 * np.sin(th)
    y_dot = u1 * np.sin(th) + u2 * np.cos(th)

    return np.array([u1_dot, u2_dot, u3_dot, th_dot, phi_dot, x_dot, y_dot])


def dynamic_model(t, q):
    global t_max, status_bar, status

    if t == 0.0:
        status_bar = tqdm(total=100, colour='magenta', position=0, leave=True)
    else:
        status_bar.update((t / t_max - status) * 100)
        status = t / t_max

    return dynamic_equations(t, q)


if __name__ == '__main__':
    u1_0 = 1.0
    u2_0 = 0.0
    u3_0 = 1.0
    th_0 = 0.0
    phi_0 = -np.pi / 2
    x_0 = 0.0
    y_0 = 0.0

    q0 = np.array([u1_0, u2_0, u3_0, th_0, phi_0, x_0, y_0])

    sol = solve_ivp(fun=dynamic_model, t_span=(0, t_max), y0=q0, method='RK45', max_step=dt, t_eval=t_sim)
    status_bar.close()
    sol.sol = [b, c, l, I1, I3, m1, m2]
    print(sol)
    with open(filename, 'wb') as f:
        pickle.dump(sol, f)
