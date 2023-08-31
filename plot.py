import numpy as np
from scipy.interpolate import interp1d
from matplotlib import patches
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'
matplotlib.rc('font', family='serif', serif='cm10', size=16)


def get_ex_data():
    x = np.arange(0, 1.05, 1 / 20)
    y1 = [0.73972019, 0.36888917, 0.4497665, 0.52831205, 0.42090202, 0.55, 0.73005815, 0.49004352, 0.245, 0.305, 0.38, 0.66000998,
          0.93799786, 0.78855405, 0.92821585, 0.87816756, 0.97225839, 0.93633211, 0.99342242, 0.9773711, 1.0]
    y2 = [0.74498084, 0.6753079, 0.9339694, 0.26502189, 0.3026732, 0.40443852, 0.44646387, 0.47971692, 0.528204, 0.62927408, 0.80889927, 0.84431554, 0.78936909,
          0.95981156, 0.6696356, 0.80937455, 0.83093156, 0.75882624, 0.84155247, 0.95338016, 1.0]

    # Define x, y, and xnew to resample at.
    xfull = np.linspace(0, 1.0, num=201, endpoint=True)
    x1start = np.linspace(0, 0.2, num=201, endpoint=True)
    xstart2 = np.linspace(0.25, 0.4, num=201, endpoint=True)
    xmid = np.linspace(0.4, 0.5, num=201, endpoint=True)
    xend = np.linspace(0.5, 1.0, num=201, endpoint=True)

    # Define interpolators.
    f1_lin = interp1d(x, y1)
    f1_cubic = interp1d(x, y1, kind='cubic')
    f2_cubic = interp1d(x, y2, kind='cubic')

    x2 = xfull
    y2 = f2_cubic(x2)

    y1start = f1_cubic(x1start)

    xboth = xmid
    yboth = f1_lin(xmid)

    x1end = np.concatenate(([0.2, 0.24, 0.25], xstart2, xmid, xend))
    y1end = np.concatenate(([0.65, 0.5, 0.55], f1_cubic(xstart2), f1_lin(xmid), f1_cubic(xend)))

    return x2, y2, x1start, y1start, x1end, y1end, xboth, yboth


def draw_step_function_ex():
    plt.vlines(x=0.2, ymin=0, ymax=0.5, color='red', zorder=2)
    plt.hlines(y=0.3, xmin=0, xmax=0.2, color='blue', zorder=2)
    plt.hlines(y=0.5, xmin=0.2, xmax=0.6, color='blue', zorder=2)
    plt.hlines(y=0.8, xmin=0.6, xmax=1.0, color='blue', zorder=2)
    plt.vlines(x=0.6, ymin=0.5, ymax=0.8, color='red', zorder=2)
    plt.vlines(x=1.0, ymin=0.8, ymax=1.0, color='red', zorder=2)
    plt.scatter([0.0, 0.2, 0.6], [0.3, 0.5, 0.8], color='blue', zorder=2)
    plt.scatter([0.2, 0.6, 1.0], [0.3, 0.5, 0.8], color='blue', facecolors='none', zorder=3)
    plt.scatter([0.2, 0.6, 1.0], [0.0, 0.5, 0.8], color='red', zorder=2)
    plt.scatter([0.2, 0.6, 1.0], [0.5, 0.8, 1.0], color='red', facecolors='none', zorder=3)

    plt.vlines(x=0.2, ymin=0.3, ymax=1.0, color='green', zorder=1, linewidth=3.0)
    plt.hlines(y=0.3, xmin=0.2, xmax=1.0, color='green', zorder=1, linewidth=3.0)
    plt.fill_between([0.2, 1.0], 0.3, 1.0, color='green', alpha=.1)
    plt.annotate(r"\textbf{G}", (0.48,0.68), fontsize=20)
    plt.annotate("$(p_1,p_2)$", (0.215, 0.325))

    plt.xlabel("$v_1$")
    plt.ylabel("$v_2$")
    plt.legend(["$f_1^*(v_2)$", "$f_2^*(v_1)$"])


def get_restrictions(x, y):
    d = {x[i]: y[i] for i in range(len(x))}
    r = 0
    R = {0.0: 0.0}
    while len(d) > 0:
        m = min(d.keys())
        if d[m] >= r:
            R[m] = d[m]
            r = d[m]
        del d[m]
    return R


def get_tail(x, y):
    last = 0
    delta = 0.1
    xs = [0.0]
    ys = [0.0]
    for i in range(len(x)):
        if y[i] - last > delta:
            xs.append(x[i])
            ys.append(last)
            xs = []
            ys = []
        else:
            xs.append(x[i])
            ys.append(y[i])
        last = y[i]
    return xs, ys


def draw_restrictions2(x, y, tail=False, color='b', linewidth=3.0):
    last = 0
    delta = 0.1
    xs = [0.0]
    ys = [0.0]
    for i in range(len(x)):
        if y[i] - last > delta:
            xs.append(x[i])
            ys.append(last)
            if not tail:
                plt.plot(xs, ys, 'g', zorder=1, linewidth=5.0)
            xs = []
            ys = []
        else:
            xs.append(x[i])
            ys.append(y[i])
        last = y[i]
    if not tail:
        return plt.plot(xs, ys, 'g', zorder=1, linewidth=5.0)
    return plt.plot(xs, ys, color, zorder=1, linewidth=linewidth)


def draw_hat2(xboth, yboth, xtail, ytail, rect=False):
    if rect:
        x = np.concatenate(([0.0, 0.2], [0.2, 0.4], xboth, [0.5, 0.6], [0.6, 0.7585918461056117], xtail))
        y = np.concatenate(([0.1, 0.1], [0.245, 0.245], yboth, [0.44, 0.44], [0.85, 0.85], ytail))
        plt.fill_between(x, y, 1, color='b', alpha=.1)

    plt.scatter([0.2, 0.5, 0.6],
                [0.245, 0.44, 0.85], color='blue', zorder=2)
    plt.scatter([0.2, 0.5, 0.6],
                [0.1, 0.38, 0.44], color='blue',
                facecolors='none', zorder=3, s=100)
    plt.plot([0.0, 0.2], [0.1, 0.1], 'b-', drawstyle='steps-post', linewidth=3.0, zorder=2)
    plt.plot([0.2, 0.4], [0.245, 0.245], 'b-', drawstyle='steps-post', linewidth=3.0, zorder=2)
    plt.plot(xboth, yboth, 'b', linewidth=3.0, zorder=2)
    plt.plot(xtail, ytail, 'b', zorder=2, linewidth=3.0)
    plt.plot([0.5, 0.6], [0.44, 0.44], 'b-', drawstyle='steps-post', linewidth=3.0, zorder=2)
    return plt.plot([0.6, 0.7585918461056117], [0.85, 0.85], 'b-', drawstyle='steps-post', linewidth=3.0, zorder=2)


def draw_star2(xboth, yboth, xtail, ytail, rect=False):
    if rect:
        x = np.concatenate(([0.0, 0.2], [0.2, 0.4], xboth, [0.5, 0.7585918461056117], xtail))
        y = np.concatenate(([0.1, 0.1], [0.245, 0.245], yboth, [0.38, 0.38], ytail))
        plt.fill_between(x, y, 1, color='b', alpha=.1)

    plt.scatter([0.2, 0.7585918461056117],
                [0.245, 0.85], color='blue', zorder=2)
    plt.scatter([0.2, 0.7585918461056117],
                [0.1, 0.38], color='blue',
                facecolors='none', zorder=3, s=100)
    plt.plot([0.0, 0.2], [0.1, 0.1], 'b-', drawstyle='steps-post', linewidth=3.0, zorder=2)
    plt.plot([0.2, 0.4], [0.245, 0.245], 'b-', drawstyle='steps-post', linewidth=3.0, zorder=2)
    plt.plot(xboth, yboth, 'b', linewidth=3.0, zorder=2)
    plt.plot(xtail, ytail, 'b', zorder=2, linewidth=3.0)
    return plt.plot([0.5, 0.7585918461056117], [0.38, 0.38], 'b-', drawstyle='steps-post', linewidth=3.0, zorder=2)


def draw_star1(xboth, yboth, xtail, ytail, rect=False):
    if rect:
        x = np.concatenate(([0.2, 0.2], xboth, [0.7585918461056117, 0.7585918461056117], xtail))
        y = np.concatenate(([0.0, 0.245], yboth, [0.38, 0.85], ytail))
        plt.fill_betweenx(y, x, 1, color='red', alpha=.1)

    plt.plot(xboth, yboth, 'r', zorder=3)
    plt.plot(xtail, ytail, 'r', zorder=3)
    plt.scatter([0.4, 0.7585918461056117], [0.245, 0.38], color='red', zorder=3)
    plt.scatter([0.2], [0.245], color='red', zorder=1, s=100)
    plt.scatter([0.5], [0.38], color='red', zorder=1, facecolor='none', s=60)
    plt.vlines(x=0.2, ymin=0.0, ymax=0.245, color='red', zorder=3)
    return plt.vlines(x=0.7585918461056117, ymin=0.38, ymax=0.85, color='red', zorder=3)

def draw_stars(rect=False):
    x2, y2, x1start, y1start, x1end, y1end, xboth, yboth = get_ex_data()
    R = get_restrictions(y2, x2)
    xtail, ytail = get_tail(list(R.keys()), list(R.values()))

    h1 = draw_star1(xboth, yboth, xtail, ytail, rect)
    h2, = draw_star2(xboth, yboth, xtail, ytail, rect)
    plt.annotate("$(p_1,p_2)$", (0.225, 0.09))

    plt.xlabel("$v_1$")
    plt.ylabel("$v_2$")
    plt.legend([h1, h2], ["$f_1^*(v_2)$", "$f_2^*(v_1)$"])

def draw_hat_and_star(rect=False):
    x2, y2, x1start, y1start, x1end, y1end, xboth, yboth = get_ex_data()
    R = get_restrictions(y2, x2)
    xtail, ytail = get_tail(list(R.keys()), list(R.values()))

    h1 = draw_star1(xboth, yboth, xtail, ytail, rect)
    h2, = draw_hat2(xboth, yboth, xtail, ytail, rect)
    plt.annotate("$(p_1,p_2)$", (0.225,0.09))

    plt.xlabel("$v_1$")
    plt.ylabel("$v_2$")
    plt.legend([h1, h2], ["$f_1^*(v_2)$", "$\hat{f}_2(v_1)$"])


def draw_restrictions1(xboth, yboth):
    plt.vlines(x=0.0, ymin=0.0, ymax=0.1, color='orange', zorder=1, linewidth=7.0)
    plt.vlines(x=0.2, ymin=0.1, ymax=0.245, color='orange', zorder=1, linewidth=7.0)
    plt.vlines(x=0.5, ymin=0.39, ymax=0.44, color='orange', zorder=1, linewidth=7.0)
    plt.plot(xboth, yboth, 'orange', linewidth=7.0, zorder=1)
    return plt.vlines(x=0.6, ymin=0.44, ymax=0.85, color='orange', zorder=1, linewidth=7.0)


def draw_f2hat_restrictions():
    x2, y2, x1start, y1start, x1end, y1end, xboth, yboth = get_ex_data()
    R = get_restrictions(y2, x2)

    handle1, = plt.plot(y2, x2, 'r', zorder=3)
    draw_restrictions2(list(R.keys()), list(R.values()), tail=True, color='orange', linewidth=7.0)
    draw_restrictions2(list(R.keys()), list(R.values()), tail=True)
    handle2, = draw_hat2(xboth, yboth)
    handler_r1 = draw_restrictions1(xboth, yboth)

    plt.xlabel("$v_1$")
    plt.ylabel("$v_2$")
    plt.legend([handle1, handle2, handler_r1], ["$f_1(v_2)$", "$\hat{f}_2(v_1)$", "$r_1^{\hat{f}_2}(v_1)$"])


def draw_random_ex_restrictions():
    x2, y2, x1start, y1start, x1end, y1end, xboth, yboth = get_ex_data()
    R = get_restrictions(y2, x2)

    handle1, = plt.plot(y2, x2, 'r', zorder=2)
    plt.plot(x1start, y1start, 'b', zorder=2)
    plt.plot(x1end, y1end, 'b', zorder=2)
    handle2, = plt.plot(xboth, yboth, 'b', zorder=2)
    handle_r2, = draw_restrictions2(list(R.keys()), list(R.values()))
    # handle2, = draw_prime2()
    # handler_r1 = draw_restrictions1()

    plt.xlabel("$v_1$")
    plt.ylabel("$v_2$")
    # plt.legend([handle1, handle_r2], ["$f_1(v_2)$", "$r_2^{f_1}(v_1)$"])
    plt.legend([handle1, handle_r2, handle2], ["$f_1(v_2)$", "$r_2^{f_1}(v_1)$", "$f_2(v_1)$"])
    # plt.legend([handle1, handle_r2, handle2, handler_r1], ["$f_1(v_2)$", "$r_2^{f_1}(v_1)$", "$f_2'(v_1)$", "$r_1^{f_2'}(v_1)$"])


def draw_random_ex():
    x2, y2, x1start, y1start, x1end, y1end, xboth, yboth = get_ex_data()

    plt.plot(y2, x2, 'r', zorder=2)
    plt.plot(x1start, y1start, 'b', linewidth=3.0, zorder=1)
    plt.plot(x1end, y1end, 'b', linewidth=3.0, zorder=1)
    plt.plot(xboth, yboth, 'b', linewidth=3.0, zorder=1)

    plt.xlabel("$v_1$")
    plt.ylabel("$v_2$")
    plt.legend(["$f_1(v_2)$", "$f_2(v_1)$"], loc='upper left')


def draw_funcs(f_1, f_2, p_1, p_2, title):
    plt.figure(figsize=(6, 6))

    x1 = np.array(list(f_1.values()))
    y1 = np.array(list(f_1.keys()))
    x2 = np.array(list(f_2.keys()))
    y2 = np.array(list(f_2.values()))

    x1_new = x1[x1 < 0.999]
    x1_new = x1[:len(x1_new)+1]
    y1_new = y1[:len(x1_new)]

    y2_new = y2[y2 < 0.999]
    y2_new = y2[:len(y2_new) + 1]
    x2_new = x2[:len(y2_new)]

    plt.plot(x1_new, y1_new, 'r-', drawstyle='steps-post', zorder=2)
    plt.plot(x2_new, y2_new, 'b-', drawstyle='steps-post', zorder=1, linewidth=3.0)

    plt.scatter([0.4], [0.3], color='black')
    plt.annotate(r"(0.4,0.3)", (0.32, 0.325))

    plt.xlabel("$v_1$")
    plt.ylabel("$v_2$")
    plt.legend(["$f_1^*(v_2)$", "$f_2^*(v_1)$"], loc='upper left')

    plt.axis('square')
    plt.grid()
    plt.tight_layout()
    plt.show()


def draw_funcs_disc(f_1, f_2):
    x1 = np.array(list(f_1.values()))
    y1 = np.array(list(f_1.keys()))
    x2 = np.array(list(f_2.keys()))
    y2 = np.array(list(f_2.values()))
    plt.plot(x1, y1, 'r-', drawstyle='steps-post')
    plt.plot(x2, y2, 'b-', drawstyle='steps-post')
    plt.plot((0.0, 1.0), (0.0, 1.0), color='green')

    plt.xlabel("$v_1$")
    plt.ylabel("$v_2$")
    plt.legend(["$f_1(v_2)$", "$f_2(v_1)$"])

    plt.axis('square')
    plt.grid()
    plt.tight_layout()
    plt.show()


def draw_matrix(G):
    plt.imshow(G, cmap='hot', interpolation='nearest')
    plt.show()


def draw_truncated():
    p1 = patches.Rectangle((0.675, 0.6), 0.05, 0.4, facecolor='magenta', alpha=0.3)
    p2 = patches.Rectangle((0.8, 0.6), 0.2, 0.4, facecolor='orange', alpha=0.3)

    ax = plt.gca()
    ax.add_patch(p1)
    ax.add_patch(p2)

    for i in np.arange(0.7, 1.1, 0.1):
        plt.scatter([i] * 6, np.arange(0.6, 1.1, 0.1), color='black')

    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel("$v_1$")
    plt.ylabel("$v_2$")


if __name__ == "__main__":
    plt.figure(figsize=(6, 6))
    draw_truncated()
    plt.axis('square')
    plt.grid()
    plt.ylim((-0.05,1.05))
    plt.xlim((-0.05,1.05))
    plt.tight_layout()
    plt.show()
