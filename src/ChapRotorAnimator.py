from ChapRotorDynamicModel import *
import cv2
import time
import imageio


def animate(f_name, capture=False, gif=True):
    global b, c, l, I1, I3, m1, m2
    h = 900
    w = 1800
    img = np.ones((h, w, 3), np.uint8) * 0
    fps = 200.0
    scale = 50
    tscale = 1
    data_points = 100
    t_step = round(fps * dt)

    if capture:
        if gif:
            frames = []
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_filename = f_name[:-4] + '.mp4'
            out = cv2.VideoWriter(video_filename, fourcc, int(1 / dt), (w, h))

    def grid(n=50):
        """
            Draw grid lines on the image.

            Args:
                n (int, optional): The spacing between grid lines. Defaults to 50.
        """
        for p in range(0, w, n):
            cv2.line(img, (p, 0), (p, h), (50, 50, 50), 1)
        for q in range(0, h, n):
            cv2.line(img, (0, q), (w, q), (50, 50, 50), 1)

    def show_fps():
        """
            Display the current frames per second (FPS) on the image.
        """
        cv2.putText(img, str(round(fps, 5)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    def show_time(t):
        """
            Display the current simulation time on the image.

            Args:
                t (float): The current simulation time.
        """
        cv2.putText(img, str(round(t, 5)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    def maintain_fps(t0_, dt_):
        """
           Maintain the desired frames per second (FPS) by waiting if necessary.

           Args:
               t0_ (float): The starting time.
               dt_ (float): The desired time step.

           Returns:
               float: The actual frames per second achieved.
       """
        while time.time() - t0_ < dt_ * tscale * t_step:
            pass
        return 1 / (time.time() - t0_)

    def keep_in_window_func(x1, y1, x2, y2):
        """
            Keep the line segment within the window by handling wrapping around.

            Args:
                x1 (int): The x-coordinate of the starting point of the line segment.
                y1 (int): The y-coordinate of the starting point of the line segment.
                x2 (int): The x-coordinate of the ending point of the line segment.
                y2 (int): The y-coordinate of the ending point of the line segment.

            Returns:
                tuple: The updated coordinates (x1, y1, x2, y2) and a flag indicating if a wrapping occurred.
        """
        x1 = x1 % w
        x2 = x2 % w
        y1 = y1 % h
        y2 = y2 % h
        jump = False
        if np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) > min(w, h) / 2:
            jump = True
        return x1, y1, x2, y2, jump

    def draw_chap_rotor(x_, y_, th_, phi_, keep_in_window=False):
        e1 = np.array([np.cos(th_), np.sin(th_), 0])
        e2 = np.array([-np.sin(th_), np.cos(th_), 0])
        C = np.array([x_, y_, 0])
        B = C - c * e1
        A = C - b * e1
        D = B + l * np.array([np.cos(phi_ + th_), np.sin(phi_ + th_), 0])

        # body
        body = np.array(
            [C + c * e1,
             C + 0.5 * c * e1 + 0.35 * c * e2,
             C + 0.35 * c * e2,
             C - 0.5 * c * e1 + 0.55 * c * e2,
             C - 1.25 * c * e1 + 0.85 * c * e2,
             A + 0.65 * c * e2,
             A - 0.4 * c * e1,
             A - 0.65 * c * e2,
             C - 1.25 * c * e1 - 0.85 * c * e2,
             C - 0.5 * c * e1 - 0.55 * c * e2,
             C - 0.35 * c * e2,
             C + 0.5 * c * e1 - 0.35 * c * e2,
             C + c * e1])

        cv2.polylines(img, np.int32(
            [np.array(
                [np.array([body[i][0] * scale + w / 2, -body[i][1] * scale + h / 2]) for i in range(len(body))])]),
                      True,
                      (255, 255, 255), 1)

        # wheels
        wheel1 = np.array(
            [A + 0.5 * c * e2 + 0.25 * c * e1,
             A + 0.5 * c * e2 - 0.25 * c * e1])
        wheel2 = np.array(
            [A - 0.5 * c * e2 + 0.25 * c * e1,
             A - 0.5 * c * e2 - 0.25 * c * e1])
        axle1 = np.array(
            [A + 0.5 * c * e2,
             A - 0.5 * c * e2])

        cv2.line(img, (int(axle1[0][0] * scale + w / 2), int(-axle1[0][1] * scale + h / 2)),
                 (int(axle1[1][0] * scale + w / 2), int(-axle1[1][1] * scale + h / 2)), (255, 0, 255), 2)
        cv2.line(img, (int(wheel1[0][0] * scale + w / 2), int(-wheel1[0][1] * scale + h / 2)),
                 (int(wheel1[1][0] * scale + w / 2), int(-wheel1[1][1] * scale + h / 2)), (50, 200, 0), scale // 10)
        cv2.line(img, (int(wheel2[0][0] * scale + w / 2), int(-wheel2[0][1] * scale + h / 2)),
                 (int(wheel2[1][0] * scale + w / 2), int(-wheel2[1][1] * scale + h / 2)), (50, 200, 0), scale // 10)

        # rotor
        rotor = np.array([B, B + (l + c) * np.array([np.cos(phi + th), np.sin(phi + th), 0])])
        cv2.line(img, (int(rotor[0][0] * scale + w / 2), int(-rotor[0][1] * scale + h / 2)),
                 (int(rotor[1][0] * scale + w / 2), int(-rotor[1][1] * scale + h / 2)), (100, 100, 100), scale // 15)

        # draw points
        cv2.circle(img, (int(A[0] * scale + w / 2), int(-A[1] * scale + h / 2)), 5, (0, 0, 255), -1)
        cv2.circle(img, (int(B[0] * scale + w / 2), int(-B[1] * scale + h / 2)), 5, (0, 255, 0), -1)
        cv2.circle(img, (int(C[0] * scale + w / 2), int(-C[1] * scale + h / 2)), 5, (255, 0, 0), -1)
        cv2.circle(img, (int(D[0] * scale + w / 2), int(-D[1] * scale + h / 2)), 5, (0, 255, 255), -1)

    sol_ = pickle.load(open(f_name, 'rb'))
    update_params(sol_.sol)

    for i in range(0, len(t_sim), t_step):
        t0 = time.time()
        u1, u2, u3, th, phi, x, y = sol_.y[:, i]

        grid(scale // 10)
        draw_chap_rotor(x, y, th, phi, True)
        # draw_velocities(x, y, x_d, y_d, 0.1)
        show_fps()
        show_time(t_sim[i])
        if capture:
            if gif:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                out.write(img)
        cv2.imshow('SnakeDynamicModel', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0)

        fps = maintain_fps(t0, dt)
        img = np.ones((h, w, 3), np.uint8) * 0

    cv2.destroyAllWindows()
    if capture:
        if gif:
            imageio.mimsave(f'{filename.split(".")[0]}.gif', frames[::4], duration=1000 / (fps * 4))
        else:
            out.release()


if __name__ == '__main__':
    animate(filename, False, False)
