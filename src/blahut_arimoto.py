import click
import numpy as np

def get_d(p_a, actions, state):
    """
    custom for the state evaluated (and results tied to deterministic env)
    p(s'|s) = - log [sum_a p(a|s)], where a are ways to get to s' from s
    """
    d = np.empty_like(p_a)
    if state == 'sl':
        d[actions['UU']] = - np.log(p_a[actions['UU']])  # 4
        d[actions['UL']] = - np.log(p_a[actions['UL']] + p_a[actions['LU']] + \
                                    p_a[actions['DU']])  # 2
        d[actions['UR']] = - np.log(p_a[actions['UR']] + p_a[actions['RU']])  # 3
        d[actions['UD']] = - np.log(p_a[actions['UD']] + p_a[actions['LL']] + \
                                    p_a[actions['LD']] + p_a[actions['RL']] + \
                                    p_a[actions['DL']] + p_a[actions['DD']])  # 0
        d[actions['LU']] = - np.log(p_a[actions['UL']] + p_a[actions['LU']] + \
                                    p_a[actions['DU']])  # 2
        d[actions['LL']] = - np.log(p_a[actions['UD']] + p_a[actions['LL']] + \
                                    p_a[actions['LD']] + p_a[actions['RL']] + \
                                    p_a[actions['DL']] + p_a[actions['DD']])  # 0
        d[actions['LR']] = - np.log(p_a[actions['LR']] + p_a[actions['RR']] + \
                                    p_a[actions['RD']] + p_a[actions['DR']])  # 1
        d[actions['LD']] = - np.log(p_a[actions['UD']] + p_a[actions['LL']] + \
                                    p_a[actions['LD']] + p_a[actions['RL']] + \
                                    p_a[actions['DL']] + p_a[actions['DD']])  # 0
        d[actions['RU']] = - np.log(p_a[actions['UR']] + p_a[actions['RU']])  # 3
        d[actions['RL']] = - np.log(p_a[actions['UD']] + p_a[actions['LL']] + \
                                    p_a[actions['LD']] + p_a[actions['RL']] + \
                                    p_a[actions['DL']] + p_a[actions['DD']])  # 0
        d[actions['RR']] = - np.log(p_a[actions['LR']] + p_a[actions['RR']] + \
                                    p_a[actions['RD']] + p_a[actions['DR']])  # 1
        d[actions['RD']] = - np.log(p_a[actions['LR']] + p_a[actions['RR']] + \
                                    p_a[actions['RD']] + p_a[actions['DR']])  # 1
        d[actions['DU']] = - np.log(p_a[actions['UL']] + p_a[actions['LU']] + \
                                    p_a[actions['DU']])  # 2
        d[actions['DR']] = - np.log(p_a[actions['LR']] + p_a[actions['RR']] + \
                                    p_a[actions['RD']] + p_a[actions['DR']])  # 1
        d[actions['DL']] = - np.log(p_a[actions['UD']] + p_a[actions['LL']] + \
                                    p_a[actions['LD']] + p_a[actions['RL']] + \
                                    p_a[actions['DL']] + p_a[actions['DD']])  # 0
        d[actions['DD']] = - np.log(p_a[actions['UD']] + p_a[actions['LL']] + \
                                    p_a[actions['LD']] + p_a[actions['RL']] + \
                                    p_a[actions['DL']] + p_a[actions['DD']])  # 0
    else:
        d[actions['UU']] = - np.log(p_a[actions['UU']])  # 11
        d[actions['UL']] = - np.log(p_a[actions['UL']])  # 7
        d[actions['UR']] = - np.log(p_a[actions['UR']] + p_a[actions['RU']])  # 8
        d[actions['UD']] = - np.log(p_a[actions['UD']] + p_a[actions['LR']] + \
                                    p_a[actions['RL']] + p_a[actions['DU']])  # 0
        d[actions['LU']] = - np.log(p_a[actions['LU']])  # 5
        d[actions['LL']] = - np.log(p_a[actions['LL']])  # 10
        d[actions['LR']] = - np.log(p_a[actions['UD']] + p_a[actions['LR']] + \
                                    p_a[actions['RL']] + p_a[actions['DU']])  # 0
        d[actions['LD']] = - np.log(p_a[actions['LD']] + p_a[actions['DL']])  # 6
        d[actions['RU']] = - np.log(p_a[actions['UR']] + p_a[actions['RU']])  # 8
        d[actions['RL']] = - np.log(p_a[actions['UD']] + p_a[actions['LR']] + \
                                    p_a[actions['RL']] + p_a[actions['DU']])  # 0
        d[actions['RR']] = - np.log(p_a[actions['RR']])  # 9
        d[actions['RD']] = - np.log(p_a[actions['RD']] + p_a[actions['DR']])  # 3
        d[actions['DU']] = - np.log(p_a[actions['UD']] + p_a[actions['LR']] + \
                                    p_a[actions['RL']] + p_a[actions['DU']])  # 0
        d[actions['DR']] = - np.log(p_a[actions['RD']] + p_a[actions['DR']])  # 3
        d[actions['DL']] = - np.log(p_a[actions['LD']] + p_a[actions['DL']])  # 6
        d[actions['DD']] = - np.log(p_a[actions['DD']])  # 4
    return d

@click.command()
@click.argument('state', type=click.Choice(['sl', 'mtl']))
def main(state):
    """
    Used to compute empowerment for lower left state of the most south cross
    """
    actions = {
        'UU': 0,
        'UL': 1,
        'UR': 2,
        'UD': 3,
        'LU': 4,
        'LL': 5,
        'LR': 6,
        'LD': 7,
        'RU': 8,
        'RL': 9,
        'RR': 10,
        'RD': 11,
        'DU': 12,
        'DR': 13,
        'DL': 14,
        'DD': 15,
    }

    p_a = np.ones(len(actions))/len(actions)  # start with uniform probs
    epsilon = 1e-4
    old_emp = -float('inf')
    d = get_d(p_a, actions, state)
    emp = np.sum(p_a * d)
    print(emp)

    count = 0
    while emp - old_emp > epsilon:
        count += 1
        z = np.sum(p_a * np.exp(d))
        p_a *= 1 / z * np.exp(d)
        old_emp = emp
        emp = np.sum(p_a * d)
        d = get_d(p_a, actions, state)
        print(emp)

    import pdb;pdb.set_trace()

if __name__ == '__main__':
    main()
