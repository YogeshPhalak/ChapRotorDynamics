import matplotlib.pyplot as plt
from ChapRotorDynamicModel import *


def plot_data(t, q, save_fig=False):
    data_name = filename.split('.')[0]
    th_plt = np.zeros(len(t))
    phi_plt = np.zeros(len(t))
    x_plt = np.zeros(len(t))
    y_plt = np.zeros(len(t))
    total_energy = np.zeros(len(t))

    for i in range(len(t)):
        u1_c, u2_c, u3_c, th_c, phi_c, x_c, y_c = q[:, i]
        th_plt[i] = th_c
        phi_plt[i] = phi_c
        x_plt[i] = x_c
        y_plt[i] = y_c

        total_energy[i] = (m1 * u1_c ** 2) / 2 + (m1 * u2_c ** 2) / 2 + (
                    m2 * (l * u2_c * np.sin(phi_c) - b * u1_c + b * l * u3_c * np.sin(phi_c)) ** 2) / (2 * b ** 2) + (
                                      I1 * u2_c ** 2) / (2 * b ** 2) + (m2 * (
                    b * u2_c - c * u2_c + l * u2_c * np.cos(phi_c) + b * l * u3_c * np.cos(phi_c)) ** 2) / (
                                      2 * b ** 2) + (I3 * (u2_c + b * u3_c) ** 2) / (2 * b ** 2) - (
                                      l ** 2 * m2 * (u2_c + b * u3_c) ** 2) / (2 * b ** 2)

    plt.plot(t, x_plt)
    plt.xlabel('Time (s)')
    plt.ylabel('x (m)')
    plt.title('x vs Time')
    if save_fig: plt.savefig(data_name + '_x.png', dpi=1000)
    plt.show()

    plt.plot(t, y_plt)
    plt.xlabel('Time (s)')
    plt.ylabel('y (m)')
    plt.title('y vs Time')
    if save_fig: plt.savefig(data_name + '_y.png', dpi=1000)
    plt.show()

    plt.plot(t, th_plt)
    plt.xlabel('Time (s)')
    plt.ylabel('theta (rad)')
    plt.title('theta vs Time')
    if save_fig: plt.savefig(data_name + '_theta.png', dpi=1000)
    plt.show()

    plt.plot(t, phi_plt)
    plt.xlabel('Time (s)')
    plt.ylabel('phi (rad)')
    plt.title('phi vs Time')
    if save_fig: plt.savefig(data_name + '_phi.png', dpi=1000)
    plt.show()

    plt.plot(x_plt, y_plt)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('y vs x')
    if save_fig: plt.savefig(data_name + '_y_vs_x.png', dpi=1000)
    plt.show()

    plt.plot(t, total_energy)
    plt.xlabel('Time (s)')
    plt.ylabel('Total Energy (J)')
    plt.title('Total Energy vs Time')
    if save_fig: plt.savefig(data_name + '_energy.png', dpi=1000)
    plt.show()


if __name__ == '__main__':
    sol = pickle.load(open(filename, 'rb'))
    update_params(sol.sol)
    q = sol.y
    t = sol.t
    plot_data(t, q, False)
