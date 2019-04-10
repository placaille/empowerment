import numpy as np
np.random.seed(123)

num_actions = 4
num_states = 5

w = np.random.rand(num_actions)
w /= w.sum()

ps_sa_all = []
for _ in range(num_actions):
    ps_sa = np.random.rand(num_states)
    ps_sa /= ps_sa.sum()
    ps_sa_all.append(ps_sa)

ps_sa_all = np.array(ps_sa_all)
gs_a = np.random.rand(*ps_sa_all.shape).T
hs_a = np.random.rand(*ps_sa_all.shape).T

first_full = 0
for a in range(num_actions):
    inner = 0
    for s_ in range(num_states):
        inner_s = ps_sa_all[a, s_] * gs_a[s_, a]
        inner += inner_s
        print('full first inner (s\', a) ({}, {}) -- {}'.format(s_, a, inner_s))

    first_full += inner * w[a]


bs_ = np.empty((num_states, num_actions))
for s_ in range(num_states):
    bs = ps_sa_all[:, s_] * gs_a[s_]
    bs_[s_] = bs
    for a in range(num_actions):
        print('vector first inner (s\', a) ({}, {}) -- {}'.format(s_, a, bs[a]))

b = bs_.sum(0)
first_vector = np.inner(w, b)


second_outer_full = 0
for a in range(num_actions):
    inner = 0
    for a_ in range(num_actions):
        for s_ in range(num_states):
            inner_s = hs_a[s_, a] * ps_sa_all[a_, s_] * w[a_]
            inner += inner_s
            print('full second inner (s\', a\', a) ({}, {}, {}) -- {}'.format(s_, a_, a, inner_s))

    second_outer_full += inner * w[a]

D_full = first_full - second_outer_full

Hs_ = np.empty((num_states, num_actions, num_actions))
for s_ in range(num_states):
    hs = np.outer(hs_a[s_], ps_sa_all[:, s_])

    for a in range(num_actions):
        for a_ in range(num_actions):
            print('vector second inner (s\', a\', a) ({}, {}, {}) -- {}'.format(s_, a_, a, hs[a_, a]))

    Hs_[s_] = hs

H = Hs_.sum(0)

second_vector = np.inner(np.inner(w, H), w)
D_vector = first_vector - second_vector

print('full: {}'.format(D_full))
print('vector: {}'.format(D_vector))
