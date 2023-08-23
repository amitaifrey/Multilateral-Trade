import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib import patches

def get_ex_data():
    x = np.arange(0, 1.05, 1 / 20)
    y1 = [0.93972019, 0.36888917, 0.8497665, 0.92831205, 0.82090202, 0.83677631, 0.93005815, 0.99004352, 0.59394092, 0.46640899, 0.77065612, 0.96000998,
          0.93799786, 0.78855405, 0.92821585, 0.87816756, 0.97225839, 0.93633211, 0.99342242, 0.9773711, 1.0]
    y2 = [0.74498084, 0.6753079, 0.9339694, 0.26502189, 0.5026732, 0.70443852, 0.34646387, 0.87971692, 0.728204, 0.62927408, 0.80889927, 0.84431554, 0.78936909,
          0.95981156, 0.7696356, 0.80937455, 0.83093156, 0.75882624, 0.84155247, 0.95338016, 1.0]
    X_Y_Spline1 = make_interp_spline(x, y1)
    X_Y_Spline2 = make_interp_spline(x, y2)
    X_ = np.linspace(x.min(), x.max(), 500)
    Y_1 = X_Y_Spline1(X_)
    Y_2 = X_Y_Spline2(X_)

    return X_, Y_1, Y_2
def draw_step_function_ex():
    plt.vlines(x=0.2, ymin=0, ymax=0.5, color='red', zorder=1)
    plt.hlines(y=0.3, xmin=0, xmax=0.2, color='blue', zorder=1)
    plt.hlines(y=0.5, xmin=0.2, xmax=0.6, color='blue', zorder=1)
    plt.hlines(y=0.8, xmin=0.6, xmax=1.0, color='blue', zorder=1)
    plt.vlines(x=0.6, ymin=0.5, ymax=0.8, color='red', zorder=1)
    plt.vlines(x=1.0, ymin=0.8, ymax=1.0, color='red', zorder=1)
    plt.scatter([0.0, 0.2, 0.6], [0.3, 0.5, 0.8], color='blue', zorder=2)
    plt.scatter([0.2, 0.6, 1.0], [0.3, 0.5, 0.8], color='blue', facecolors='none', zorder=3)
    plt.scatter([0.2, 0.6, 1.0], [0.0, 0.5, 0.8], color='red', zorder=2)
    plt.scatter([0.2, 0.6, 1.0], [0.5, 0.8, 1.0], color='red', facecolors='none', zorder=3)

    plt.xlabel("$v_1$", fontsize=14)
    plt.ylabel("$v_2$", fontsize=14)
    plt.title("Example of Step Functions for a Mechanism", fontsize=18, y=1.01)
    plt.legend(["$f_1(v_2)$", "$f_2(v_1)$"])

    plt.grid()
    plt.show()

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

def draw_restrictions2(x, y):
    last = 0
    delta = 0.1
    xs = [0.0]
    ys = [0.0]
    for i in range(len(x)):
        if y[i] - last > delta:
            xs.append(x[i])
            ys.append(last)
            plt.plot(xs, ys, 'g')
            xs = []
            ys = []
        else:
            xs.append(x[i])
            ys.append(y[i])
        last = y[i]
    h = plt.plot(xs, ys, 'g')
    plt.plot(xs, ys, 'g')
    return h

def draw_star2(rect=False):
    if rect:
        x = [0.0, 0.22796545219495468, 0.22796545219495468, 0.3431373136961936, 0.3431373136961936, 0.617639662274638, 0.617639662274638, 0.7585918461056117, 0.7585918461056117]
        y = [0.1, 0.1, 0.25, 0.25, 0.4, 0.4, 0.8, 0.8, 0.854809619238477]
        plt.fill_between(x, y, 1, color='blue', alpha=.1)

    plt.scatter([0.22796545219495468, 0.3431373136961936, 0.617639662274638, 0.7585918461056117], [0.25, 0.4, 0.8, 0.854809619238477], color='blue', zorder=2)
    plt.scatter([0.22796545219495468, 0.3431373136961936, 0.617639662274638, 0.7585918461056117], [0.1, 0.25, 0.4, 0.8], color='blue',
                facecolors='none', zorder=3)
    plt.plot([0.0, 0.22796545219495468], [0.1, 0.1], 'b-', drawstyle='steps-post')
    plt.plot([0.2280251270926629, 0.3431373136961936], [0.25, 0.25], 'b-', drawstyle='steps-post')
    plt.plot([0.3444122023091325, 0.617639662274638], [0.4, 0.4], 'b-', drawstyle='steps-post')
    return plt.plot([0.6179681202052336, 0.7585918461056117], [0.8, 0.8], 'b-', drawstyle='steps-post')

def draw_final2(rect=False):
    if rect:
        x = [0.0, 0.22796545219495468, 0.22796545219495468, 0.4, 0.4, 0.7585918461056117, 0.7585918461056117]
        y = [0.1, 0.1, 0.25, 0.25, 0.4, 0.4, 0.8]
        plt.fill_between(x, y, 1, color='blue', alpha=.1)

    plt.scatter([0.22796545219495468, 0.4, 0.7585918461056117], [0.25, 0.4, 0.854809619238477], color='blue', zorder=2)
    plt.scatter([0.22796545219495468, 0.4, 0.7585918461056117], [0.1, 0.25, 0.4], color='blue',
                facecolors='none', zorder=3)
    plt.plot([0.0, 0.22796545219495468], [0.1, 0.1], 'b-', drawstyle='steps-post')
    plt.plot([0.2280251270926629, 0.4], [0.25, 0.25], 'b-', drawstyle='steps-post')
    return plt.plot([0.4, 0.7585918461056117], [0.4, 0.4], 'b-', drawstyle='steps-post')

def draw_star1(rect=False):
    if rect:
        y = [0.0, 0.25, 0.25, 0.4, 0.4, 0.854809619238477, 0.854809619238477]
        x = [0.22796545219495468, 0.22796545219495468, 0.4, 0.4, 0.7585918461056117, 0.7585918461056117, 0.8]
        plt.fill_betweenx(y, x, 1, color='red', alpha=.1)

    plt.scatter([0.4, 0.7585918461056117], [0.25, 0.4], color='red', zorder=2)
    plt.scatter([0.22796545219495468, 0.4], [0.25, 0.4], color='red',
                facecolors='none', zorder=3)
    plt.vlines(x=0.22796545219495468, ymin=0.0, ymax=0.25, color='red', zorder=1)
    plt.vlines(x=0.4, ymin=0.25, ymax=0.4, color='red', zorder=1)
    return plt.vlines(x=0.7585918461056117, ymin=0.4, ymax=0.854809619238477, color='red', zorder=1)

def draw_stars_end(x, y, rect=False):
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
    plt.plot(xs, ys, 'magenta')
    if rect:
        plt.fill_between(xs, ys, 1, color='blue', alpha=.1)
        plt.fill_betweenx(ys, xs, 1, color='red', alpha=.1)

def draw_final(rect=False):
    X_, Y_1, Y_2 = get_ex_data()
    R = get_restrictions(Y_2, X_)

    draw_stars_end(list(R.keys()), list(R.values()), rect)
    h1 = draw_star1(rect)
    h2, = draw_final2(rect)

    plt.xlabel("$v_1$", fontsize=14)
    plt.ylabel("$v_2$", fontsize=14)
    plt.legend([h1, h2], ["$f_1^*(v_2)$", "$f_2^*(v_1)$"])
    plt.grid()
    plt.show()
def draw_stars(rect=False):
    X_, Y_1, Y_2 = get_ex_data()
    R = get_restrictions(Y_2, X_)

    draw_stars_end(list(R.keys()), list(R.values()), rect)
    h1 = draw_star1(rect)
    h2, = draw_star2(rect)

    plt.xlabel("$v_1$", fontsize=14)
    plt.ylabel("$v_2$", fontsize=14)
    plt.legend([h1, h2], ["$f_1^*(v_2)$", "$f_2'(v_1)$"])
    plt.grid()
    plt.show()

def draw_restrictions1():
    plt.scatter([0.0, 0.22796545219495468, 0.3431373136961936, 0.617639662274638],[0.1, 0.25, 0.4, 0.8], color='orange',
                facecolors='none', zorder=3)
    plt.scatter([0.22796545219495468, 0.3431373136961936, 0.617639662274638, 0.7585918461056117], [0.1, 0.25, 0.4, 0.8], color='orange',
                zorder=2)
    plt.vlines(x=0.0, ymin=0.0, ymax=0.1, color='orange', zorder=1)
    plt.vlines(x=0.22796545219495468, ymin=0.1, ymax=0.25, color='orange', zorder=1)
    plt.vlines(x=0.3431373136961936, ymin=0.25, ymax=0.4, color='orange', zorder=1)
    plt.vlines(x=0.617639662274638, ymin=0.4, ymax=0.8, color='orange', zorder=1)
    return plt.vlines(x=0.7585918461056117, ymin=0.8, ymax=0.854809619238477, color='orange', zorder=1)

def draw_random_ex_restrictions():
    X_, Y_1, Y_2 = get_ex_data()
    R = get_restrictions(Y_2, X_)

    handle1, = plt.plot(Y_2, X_, 'r')
    handle_r2, = draw_restrictions2(list(R.keys()), list(R.values()))
    #handle2, = draw_star2()
    #handler_r1 = draw_restrictions1()

    plt.xlabel("$v_1$", fontsize=14)
    plt.ylabel("$v_2$", fontsize=14)
    plt.legend([handle1, handle_r2], ["$f_1(v_2)$", "$r_2^{f_1}(v_1)$"])
    #plt.legend([handle1, handle_r2, handle2, handler_r1], ["$f_1(v_2)$", "$r_2^{f_1}(v_1)$", "$f_2'(v_1)$", "$r_1^{f_2'}(v_1)$"])
    plt.grid()
    plt.show()

def draw_random_ex():
    X_, Y_1, Y_2 = get_ex_data()

    plt.plot(Y_2, X_, 'r')
    plt.plot(X_, Y_1, 'b')
    plt.xlabel("$v_1$", fontsize=14)
    plt.ylabel("$v_2$", fontsize=14)
    plt.legend(["$f_1(v_2)$", "$f_2(v_1)$"])

    plt.grid()
    plt.show()

def draw_funcs(f_1, f_2, p_1, p_2, title):
    plt.figure(figsize=(6, 6))

    x1 = np.array(list(f_1.values()))
    y1 = np.array(list(f_1.keys()))
    x2 = np.array(list(f_2.keys()))
    y2 = np.array(list(f_2.values()))
    plt.plot(x1[:p_2], y1[:p_2], 'r-', drawstyle='steps-post')
    plt.plot(x2[:p_1], y2[:p_1], 'b-', drawstyle='steps-post')
    plt.plot(x1[p_2:], y1[p_2:], 'm-', drawstyle='steps-post')
    plt.plot(x2[p_1:], y2[p_1:], 'm-', drawstyle='steps-post')

    plt.xlabel("$v_1$", fontsize=14)
    plt.ylabel("$v_2$", fontsize=14)
    plt.title(title, fontsize=16, y=1.01)
    plt.legend(["$f_1(v_2)$", "$f_2(v_1)$"])
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

    plt.xlabel("$v_1$", fontsize=14)
    plt.ylabel("$v_2$", fontsize=14)
    plt.legend(["$f_1(v_2)$", "$f_2(v_1)$"])

    plt.grid()
    plt.show()

def draw_matrix(G):
    plt.imshow(G, cmap='hot', interpolation='nearest')
    plt.show()

def draw_truncated():
    p1 = patches.Rectangle((0.375, 0.3), 0.05, 0.7, facecolor='g', alpha=0.3)
    p2 = patches.Rectangle((0.5, 0.3), 0.5, 0.7, facecolor='orange', alpha=0.3)

    ax = plt.gca()
    ax.add_patch(p1)
    ax.add_patch(p2)

    for i in np.arange(0.4, 1.1, 0.1):
        plt.scatter([i] * 8, np.arange(0.3, 1.1, 0.1), color='black')

    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))

    plt.grid()
    plt.show()

if __name__ == "__main__":
    draw_final()