from ChapRotorDynamicModel import *
import pickle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def energy(q_):
    u1_, u2_, u3_, th_, phi_, x_, y_ = q_
    return (m1 * u1_ ** 2) / 2 + (m1 * u2_ ** 2) / 2 + (
            m2 * (l * u2_ * np.sin(phi_) - b * u1_ + b * l * u3_ * np.sin(phi_)) ** 2) / (2 * b ** 2) + (
            I1 * u2_ ** 2) / (2 * b ** 2) + (
            m2 * (b * u2_ - c * u2_ + l * u2_ * np.cos(phi_) + b * l * u3_ * np.cos(phi_)) ** 2) / (2 * b ** 2) + (
            I3 * (u2_ + b * u3_) ** 2) / (2 * b ** 2) - (l ** 2 * m2 * (u2_ + b * u3_) ** 2) / (2 * b ** 2)


def find_datapoints(E):
    file = open('trajectory_data_1000.pkl', 'wb')
    for v1 in np.linspace(-np.pi / 2, np.pi / 2, 20):
        for v2 in np.linspace(-np.pi / 2, np.pi / 2, 20):
            for v3 in np.linspace(-np.pi / 2, np.pi / 2, 20):
                for phi in np.linspace(-np.pi / 2, np.pi / 2, 20):
                    E_ = energy([b * v1, b * v2, v3, v1, phi, 0, 0])
                    if abs(E_ - E) < 0.05:
                        print()
                        print('E = ', E_)
                        print('v1 = ', v1)
                        print('v2 = ', v2)
                        print('v3 = ', v3)
                        print('phi = ', phi)
                        print()
                        q0 = [b * v1, b * v2, v3, v1, phi, 0, 0]
                        sol = solve_ivp(fun=dynamic_model, t_span=(0, t_max), y0=q0, method='RK45', max_step=dt,
                                        t_eval=t_sim)
                        # status_bar.close()
                        status = 0.0
                        sol.sol = [b, c, l, I1, I3, m1, m2]
                        print(sol)
                        pickle.dump(sol, file)

    file.close()


def plot_data_(t):
    # open data file and get v1 v2 v3 from all the data points
    file = open('trajectory_data_1000.pkl', 'rb')
    v1 = np.zeros(len(t))
    v2 = np.zeros(len(t))
    v3 = np.zeros(len(t))

    # 3d plot v1 v2 v3
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('v1')
    ax.set_ylabel('v2')
    ax.set_zlabel('v3')

    while True:
        try:
            sol = pickle.load(file)
            for i in range(len(t)):
                v1[i] = sol.y[0, i] / b
                v2[i] = sol.y[1, i] / b
                v3[i] = sol.y[2, i]
            ax.plot(v1, v2, v3)
            plt.ion()

        except Exception:
            plt.ioff()
            plt.show()
            break


if __name__ == '__main__':
    # find_datapoints(3.0)
    plot_data_(t_sim)
