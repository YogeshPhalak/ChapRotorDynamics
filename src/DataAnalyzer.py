import matplotlib.pyplot as plt
from ChapRotorDynamicModel import *


def plot_data(t, q, save_fig=False):
    data_name = filename.split('.')[0]
    th_plt = np.zeros(len(t))
    phi_plt = np.zeros(len(t))
    x_plt = np.zeros(len(t))
    y_plt = np.zeros(len(t))

    for i in range(len(t)):
        u1_c, u2_c, u3_c, th_c, phi_c, x_c, y_c = q[:, i]
        th_plt[i] = th_c
        phi_plt[i] = phi_c
        x_plt[i] = x_c
        y_plt[i] = y_c

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


if __name__ == '__main__':
    sol = pickle.load(open(filename, 'rb'))
    update_params(sol.sol)
    q = sol.y
    t = sol.t
    plot_data(t, q, False)
