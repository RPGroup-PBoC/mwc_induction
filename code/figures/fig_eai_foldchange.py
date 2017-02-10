import numpy as np
import matplotlib.pyplot as plt
import mwc_induction_utils as mwc
mwc.set_plotting_style()

R = [22, 60, 120, 260, 880, 1220, 1740]
ka = 139E-6
ki = 0.53E-6
ep_ra = -13.9
ep_ai = np.linspace(-10, 10, 500)

def ep_ai_var(R, ka, ki, ep_ra, ep_ai):
    return (1 + (1 / (1 + np.exp(-ep_ai))*(R / 4.6E6)*np.exp(-ep_ra)))**(-1)


plt.figure()
for r in R:
    fc = ep_ai_var(r, ka, ki, ep_ra, ep_ai)
    plt.plot(ep_ai, fc)
plt.xlabel(r'$\Delta\varepsilon_{AI}\, k_BT$')
plt.ylabel('fold-change')
plt.show()
