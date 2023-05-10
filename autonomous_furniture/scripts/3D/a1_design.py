import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    kappa = 1.0000000
    k = 1e-2

    d_max = 3.0
    d_min = 0.0
    N = 1000
    alpha = 1.5

    kappas = np.linspace(1, 10, N)
    d = np.linspace(d_max, d_min, N)
    w1 = np.copy(d)
    w1_alternative = np.copy(w1)
    new_term = np.copy(w1_alternative)

    r = d / (d + k)
    w1 = 1 / 2 * (1 + np.tanh(kappa * (d - alpha))) * r
    w1_alternative = (
        1
        / 2
        * (1 + np.tanh(kappa * (d - alpha)))
        * r
        * (kappa - 1)
        / (kappa - 1 + 1e-6)
    )
    new_term = (kappas - 1) / (kappas - 1 + 1e-6)

    plt.plot(-d, w1, -d, w1_alternative)
    plt.show()

    plt.plot(kappas, new_term)
    plt.show()
