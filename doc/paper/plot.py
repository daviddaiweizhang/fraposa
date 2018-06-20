import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

methods = ['sp', 'ap', 'oadp', 'adp']
n = np.arange(5, dtype=int) * 500 + 1000

ggsim = np.loadtxt('ggsim_runtimes.dat')
for i in range(4):
    plt.plot(n, ggsim[i], label=methods[i])
plt.xticks(n)
plt.legend()
plt.savefig('img/ggsim_runtimes.png', dpi=300)
plt.close('all')

ggsim = np.loadtxt('ggsim_errors.dat')
for i in range(4):
    plt.plot(n, ggsim[i], label=methods[i])
plt.xticks(n)
plt.legend()
plt.savefig('img/ggsim_errors.png', dpi=300)
plt.close('all')

