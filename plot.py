import matplotlib.pyplot as plt
import numpy as np

pcs_stu_proj = np.loadtxt('pcs_stu_proj.dat')
pcs_stu_hdpca = np.loadtxt('pcs_stu_hdpca.dat')
pcs_stu_onl = np.loadtxt('pcs_stu_onl.dat')
pcs_stu_trace = np.loadtxt('pcs_stu_trace.dat')

plt.plot(pcs_stu_proj[:, 0], pcs_stu_proj[:, 1], 'o', alpha=0.3, label='projection')
plt.plot(pcs_stu_hdpca[:, 0], pcs_stu_hdpca[:, 1], 'o', alpha=0.3, label='hdpca')
plt.plot(pcs_stu_onl[:, 0], pcs_stu_onl[:, 1], 'o', alpha=0.3, label='online')
plt.plot(pcs_stu_trace[:, 0], pcs_stu_trace[:, 1], 'o', alpha=0.3, label='trace')
plt.legend()
plt.show()
