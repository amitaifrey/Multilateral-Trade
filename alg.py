import numpy as np
import tqdm
import plot

SAMPLE_SIZE = 4000
MIN_PROB = 1 / SAMPLE_SIZE


def init_buyer_arrays(S, P_b):
    C = 1 - np.cumsum(P_b)
    C = np.concatenate(([1], C[:-1]))
    argmax = np.argmax(C <= MIN_PROB)
    E = np.flip(np.cumsum(np.flip(P_b[:argmax] * S[:argmax]))) / C[:argmax]
    E = np.concatenate((E, [0] * (len(S) - argmax)))
    return C, E


def init_seller_arrays(S, P_s):
    C = np.cumsum(P_s)
    argmin = np.argmin(C <= MIN_PROB)
    E = np.cumsum(P_s[argmin:] * S[argmin:]) / C[argmin:]
    E = np.concatenate(([0] * argmin, E))
    return C, E


def get_best_price(S, C_s, C_b, E_s, E_b):
    max_price, max_gft = 0, -1
    for i in range(1, len(S)):
        gft = (E_b[i] - E_s[i]) * C_b[i] * C_s[i]
        if gft >= max_gft:
            max_price, max_gft = i, gft
    return max_price


def init_matrix(S, P_1, P_2, C_s, E_s):
    G = np.zeros((len(S), len(S)))
    G[-1, -1] = P_1[-1] * P_2[-1] * (S[-1] - E_s[-1]) * C_s[-1]
    # for i in range(len(S) - 2, 0, -1):
    #     G[i, -1] = P_1[-1] * P_2[i] * (S[-1] - E_s[-1]) + G[i + 1, -1]
    #     G[-1, i] = P_1[i] * P_2[-1] * (S[-1] - E_s[-1]) + G[-1, i + 1]
    return G


def fill_matrix(S, P_1, P_2, C_s, C_1, C_2, E_s, E_1, E_2, G, p_1, p_2):
    for i in tqdm.tqdm(range(len(S) - 1, p_1 - 1, -1)):
        for j in range(len(S) - 1, p_2 - 1, -1):
            G_1 = 0
            G_2 = 0
            if i < len(S) - 1:
                G_1 = P_1[i] * C_2[j] * (E_2[j] - E_s[j]) * C_s[j] + G[i + 1, j]
            if j < len(S) - 1:
                G_2 = P_2[j] * C_1[i] * (E_1[i] - E_s[i]) * C_s[i] + G[i, j + 1]
            G[i, j] = max(G_1, G_2)


def find_funcs(G, P_1, P_2, C_s, C_1, C_2, E_s, E_1, E_2, S, p_1, p_2):
    f_1, f_2 = {}, {}
    for j in range(p_2):
        f_1[S[j]] = S[p_1]
    for i in range(p_1):
        f_2[S[i]] = S[p_2]

    i, j = p_1, p_2
    while i <= len(S) - 1 and j <= len(S) - 1:
        if S[j] not in f_1:
            f_1[S[j]] = S[i]
        f_2[S[i]] = S[j]

        if i == len(S) - 1:
            j = j + 1
        elif j == len(S) - 1:
            i = i + 1
        elif G[i, j] == P_1[i] * C_2[j] * (E_2[j] - E_s[j]) * C_s[j] + G[i + 1, j]:
            i = i + 1
        else:
            j = j + 1
    return f_1, f_2


def run(S, P_s, P_1, P_2):
    C_1, E_1 = init_buyer_arrays(S, P_1)
    C_2, E_2 = init_buyer_arrays(S, P_2)
    C_s, E_s = init_seller_arrays(S, P_s)
    p_1 = get_best_price(S, C_s, C_1, E_s, E_1)
    p_2 = get_best_price(S, C_s, C_2, E_s, E_2)
    G = init_matrix(S, P_1, P_2, C_s, E_s)
    fill_matrix(S, P_1, P_2, C_s, C_1, C_2, E_s, E_1, E_2, G, p_1, p_2)
    f_1, f_2 = find_funcs(G, P_1, P_2, C_s, C_1, C_2, E_s, E_1, E_2, S, p_1, p_2)
    return f_1, f_2, G, p_1, p_2


def fixed_prices(v_s, v_1, v_2, p1, p2):
    if v_1 >= p1 and p1 >= v_s:
        return v_1 - v_s
    if v_2 >= p2 and p2 >= v_s:
        return v_2 - v_s
    return 0


# def array_mech_approx(v_s, v_1, v_2, f_1, f_2, S):
#     v_1s = S[np.argmin(S < v_1) - 1]
#     v_2s = S[np.argmax(S >= v_2)]
#     if v_1 == f_1[v_2s] and v_2 == f_2[v_1s] and f_1[v_2s] >= v_s and f_2[v_1s] >= v_s:
#         return max(v_1, v_2) - v_s
#     if v_1 >= f_1[v_2s] and f_1[v_2s] >= v_s:
#         return v_1 - v_s
#     if v_2 >= f_2[v_1s] and f_2[v_1s] >= v_s:
#         return v_2 - v_s
#     return 0

def array_mech(v_s, v_1, v_2, f_1, f_2):
    if v_1 > f_1[v_2] and f_1[v_2] >= v_s:
        return v_1 - v_s
    if v_2 > f_2[v_1] and f_2[v_1] >= v_s:
        return v_2 - v_s
    if v_1 == f_1[v_2] and v_2 == f_2[v_1] and f_1[v_2] >= v_s and f_2[v_1] >= v_s:
        return max(v_1, v_2) - v_s
    if v_1 == f_1[v_2] and f_1[v_2] >= v_s:
        return v_1 - v_s
    if v_2 == f_2[v_1] and f_2[v_1] >= v_s:
        return v_2 - v_s
    return 0


# def calc_random_gft(f_1, f_2, S):
#     T = 100000
#     opt = 0.0
#     learned = 0.0
#     for i in tqdm.trange(T):
#         vs = np.random.uniform(0, 1)
#         v1 = np.random.uniform(0, 1)
#         v2 = np.random.uniform(0, 1.0 / 2.0)
#         opt_gft = fixed_prices(vs, v1, v2, max(0.5, v2*0.5 + 3.0/8.0), max(0.25, v1*2 - 0.75))
#         learned_gft = array_mech_approx(vs, v1, v2, f_1, f_2, S)
#         opt += opt_gft
#         learned += learned_gft
#     print(opt, learned)
#     print('OPT: %f\n MECH: %f\n PERC: %f\n' % (opt / T, learned / T, learned / opt))

def calc_gft(f_1, f_2, g_1, g_2, S, P_s, P_1, P_2):
    gft = 0
    for i_s in range(len(S)):
        if P_s[i_s] == 0:
            continue
        for i_1 in range(len(S)):
            if P_1[i_1] == 0:
                continue
            for i_2 in range(len(S)):
                if P_2[i_2] == 0:
                    continue
                vs, v1, v2 = S[i_s], S[i_1], S[i_2]
                gft += array_mech(vs, v1, v2, f_1, f_2) * P_s[i_s] * P_1[i_1] * P_2[i_2]
                if array_mech(vs, v1, v2, f_1, f_2) != array_mech(vs, v1, v2, g_1, g_2):
                    print(vs, v1, v2, f_1[v2], f_2[v1], g_1[v2], g_2[v1])
    return gft


def sample_dists():
    S_s = np.sort(np.random.uniform(0, 1, SAMPLE_SIZE))
    S_1 = np.sort(np.random.uniform(0, 1, SAMPLE_SIZE))
    S_2 = np.sort(np.random.uniform(0, 1, SAMPLE_SIZE) / 2.0)
    S = np.sort(np.unique(np.concatenate((S_s, S_1, S_2, [0, 1]))))
    P_s = np.where(np.isin(S, S_s), 1 / SAMPLE_SIZE, 0)
    P_1 = np.where(np.isin(S, S_1), 1 / SAMPLE_SIZE, 0)
    P_2 = np.where(np.isin(S, S_2), 1 / SAMPLE_SIZE, 0)
    return S, P_s, P_1, P_2


def create_dists_grid():
    step_size = 1.0 / SAMPLE_SIZE
    S_s = np.arange(0.0, 1.0, step_size)
    S_1 = np.arange(0.0, 1.0, step_size)
    S_2 = np.arange(0.0, 0.5, step_size)
    #S_1 = np.concatenate((np.arange(0.0, 0.5, step_size*3), np.arange(0.5, 3.0/4.0, step_size)))
    #S_2 = np.concatenate((np.arange(0.0, 0.25, step_size), np.arange(0.5, 1.0, step_size)))
    S = np.sort(np.unique(np.concatenate((S_s, S_1, S_2, [0, 1]))))
    P_s = np.where(np.isin(S, S_s), 1 / SAMPLE_SIZE, 0)
    P_1 = np.where(np.isin(S, S_1), 1 / SAMPLE_SIZE, 0)
    P_1 = P_1 / np.sum(P_1)
    P_2 = np.where(np.isin(S, S_2), 1 / SAMPLE_SIZE, 0)
    P_2 = P_2 / np.sum(P_2)
    return S, P_s, P_1, P_2


def create_dists_disc():
    # S_s = {k: 1.0 / SAMPLE_SIZE for k in np.arange(0.2, 0.7, 0.5 / SAMPLE_SIZE)}
    S_s = {0.0: 0.125, 0.25: 0.125, 0.5: 0.125, 0.75: 0.125, 0.125: 0.125, 0.375: 0.125, 0.625: 0.125, 0.875: 0.125}
    S_1 = {0.0: 0.25, 0.25: 0.25, 0.5: 0.25, 0.75: 0.25}
    # S_1 = S_1 | {k: 0.25 / SAMPLE_SIZE for k in np.arange(0.6, 0.8, 0.2 / SAMPLE_SIZE)}
    S_2 = {0.0: 0.0, 0.125: 0.25, 0.375: 0.25, 0.625: 0.25, 0.875: 0.25}
    # S_2 = S_2 | {k: 0.25 / SAMPLE_SIZE for k in np.arange(0.6, 0.8, 0.2 / SAMPLE_SIZE)}
    S = np.sort(np.unique(np.concatenate((list(S_s.keys()), list(S_1.keys()), list(S_2.keys()), [0, 1]))))
    P_s = [S_s[k] if k in S_s else 0.0 for k in S]
    P_s = P_s / np.sum(P_s)
    P_1 = [S_1[k] if k in S_1 else 0.0 for k in S]
    P_1 = P_1 / np.sum(P_1)
    P_2 = [S_2[k] if k in S_2 else 0.0 for k in S]
    P_2 = P_2 / np.sum(P_2)
    print(np.sum(P_s), np.sum(P_1), np.sum(P_2))
    return S, P_s, P_1, P_2


def clean_func(S, P, f):
    g = {}
    for i in range(len(S)):
        if P[i] != 0:
            g[S[i]] = f[S[i]]
    return g


def main():
    S, P_s, P_1, P_2 = create_dists_grid()
    f_1, f_2, G, p_1, p_2 = run(S, P_s, P_1, P_2)
    # draw_matrix(G)
    title = r"$v_s\sim U[0,1], v_1\sim U[0,1], v_2\sim U[0,\frac{1}{2}]$"
    #title = r"$v_s\sim U[0,1], v_1\sim \begin{cases} U[0,\frac{1}{2}], & \frac{1}{4} \\ U[\frac{1}{2}, \frac{3}{4}], & \frac{3}{4} \end{cases}, v_1\sim \begin{cases} U[0,\frac{1}{4}], & \frac{3}{4} \\ U[\frac{1}{2}, 1], & \frac{1}{4} \end{cases}$\\"
    plot.draw_funcs(f_1, f_2, p_1, p_2, title)
    #plot.draw_funcs_disc(f_1, f_2)
    #print(f_1, f_2)


if __name__ == "__main__":
    main()
