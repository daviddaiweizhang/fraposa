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
plt.title('Runtime', fontsize=20)
fig = plt.gcf()
fig.set_size_inches((12,5))
plt.savefig('img/ggsim_runtimes.png', dpi=300)
plt.close('all')

ggsim = np.loadtxt('ggsim_errors_adp.dat')
for i in range(4):
    plt.plot(n, ggsim[i], label=methods[i])
plt.xticks(n)
plt.legend()
plt.title('Error (from ADP)', fontsize=20)
fig = plt.gcf()
fig.set_size_inches((12,5))
plt.savefig('img/ggsim_errors_adp.png', dpi=300)
plt.close('all')

ggsim = np.loadtxt('ggsim_errors_ctr.dat')
for i in range(4):
    plt.plot(n, ggsim[i], label=methods[i])
plt.xticks(n)
plt.legend()
plt.title('Error (between centers)', fontsize=20)
fig = plt.gcf()
fig.set_size_inches((12,5))
plt.savefig('img/ggsim_errors_ctr.png', dpi=300)
plt.close('all')

