import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    kappa_list = [1.0, 1.5, 2.0, 2.5]
    k = 1e-2

    d_max = 3.0
    d_min = 0.0
    N = 1000
    alpha = 1.5

    kappas = np.linspace(1, 1.1, N)
    d = np.linspace(d_max, d_min, N)
    # w1 = np.copy(d)
    # w1_alternative = np.copy(w1)
    # new_term = np.copy(w1_alternative)

    for i in range(len(kappa_list)):
        r = d / (d + k)
        w1 = 1 / 2 * (1 + np.tanh(kappa_list[i] * (d - alpha))) * r
        w1_alternative = (
            1
            / 2
            * (1 + np.tanh(kappa_list[i] * (d - alpha)))
            * r
            * (kappa_list[i] - 1)
            / (kappa_list[i] - 1 + 1e-6)
        )

        # plt.plot(-d, w1, label="$\mu=$"+str(kappa_list[i]))
        plt.plot(-d, w1_alternative, label="$\mu=$" + str(kappa_list[i]))

    new_term = (kappas - 1) / (kappas - 1 + 1e-6)

    plt.legend()
    plt.xlabel("d [m]")
    plt.ylabel("a1 [-]")
    plt.title("a1 new")
    plt.show()

    # plt.plot(kappas, new_term)
    # plt.xlabel("$\mu$")
    # plt.ylabel("$\dfrac{\mu - 1}{\mu - 1 + 10^{-6}}$")
    plt.show()
