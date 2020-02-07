# Chargement de dépendances

import numpy as np
import matplotlib.pyplot as plt
from math import *

# Discrétisation
A = 0
B = 500
N = 101  # Nombre de points de discrétisation
Delta = (B-A)/(N-1)
discretization_indexes = np.arange(N)
discretization = discretization_indexes*Delta
# Paramètres du modèle

mu = -5
a = 50
sigma2 = 12

# Données

observation_indexes = [0, 20, 40, 60, 80, 100]
depth = np.array([0, -4, -12.8, -1, -6.5, 0])

# Indices des composantes correspondant aux observations et aux componsantes non observées

unknown_indexes = list(set(discretization_indexes)-set(observation_indexes))


# Question 1
def c(h, a, sig2):
    try:
        return sig2*exp(-abs(h)/a)
    except ZeroDivisionError as identifier:
        print("La valeur de a doit être différente que 0.")


def mat_dis(dis):
    mat = np.zeros((len(dis), len(dis)))
    for i in range(len(dis)):
        for j in range(len(dis)):
            mat[i, j] = abs(dis[i] - dis[j])
    # print(mat)
    return mat


def mat_cov(dis, f, a, sig2):
    mat = np.zeros(dis.shape)
    for i in range(dis.shape[0]):
        for j in range(dis.shape[1]):
            mat[i, j] = f(dis[i, j], a, sig2)
    print(mat)
    return mat


# Question 2
mat_distance = mat_dis(discretization)

# Question 3
mat_covariance = mat_cov(mat_distance, c, a, sigma2)

# Question 4
# Entre les observations
cov_22 = mat_covariance[observation_indexes, :]
cov_22 = cov_22[:, observation_indexes]
# Entre les incunnues
cov_11 = mat_covariance[unknown_indexes, :]
cov_11 = cov_11[:, unknown_indexes]
# Entre les observations et les inconnues
cov_21 = mat_covariance[observation_indexes, :]
cov_21 = cov_21[:, unknown_indexes]
cov_12 = cov_21.T

# Question 5
esp = (np.ones(N)*mu).T
# Espérance conditionnelle
cov_22_inv = np.linalg.inv(cov_22)
esp_con = esp[unknown_indexes]+cov_12.dot(cov_22_inv).dot(depth.T-esp[observation_indexes])
# ICI si '-' remplacé par '+', esp_con valeurs sont normales
print(esp_con.shape)

# Question 6
# Matrice de covariance
cov_cdt = cov_11-cov_12.dot(cov_22_inv).dot(cov_21)
print(cov_cdt.shape)
cov_cdt_dgnl = [cov_cdt[i, i] for i in range(len(unknown_indexes))]
print(len(cov_cdt_dgnl))
unknown_distance = np.array(unknown_indexes)*Delta
known_distance = np.array(observation_indexes)*Delta

# Plot
plt.plot(unknown_distance, cov_cdt_dgnl)
plt.title('Diagonale covariance')
plt.show()

# Question 7


def fusion_2(c, unc, rel, sim):
    dis = np.zeros(len(c)+len(unc))
    dep = np.zeros(len(c)+len(unc))
    for i in range(len(c)):
        dis[c[i]] = c[i]
        dep[c[i]] = rel[i]
    for i in range(len(unc)):
        dis[unc[i]] = unc[i]
        dep[unc[i]] = sim[i]
    return dis, dep


x = np.random.multivariate_normal(esp_con, cov_cdt).T
# plt.plot(unknown_distance, x)
sim = fusion_2(observation_indexes, unknown_indexes, depth, x)
plt.plot(sim[0]*Delta, sim[1])
plt.title('A single simulation')
plt.show()


# Question 8
def longueur(depth, delta):
    res = 0
    for i in range(len(depth)-1):
        res += sqrt(delta**2+(depth[i]-depth[i+1])**2)
    return res


# Question 9
def simulation(esp, cov, delta, num=1):
    x = np.random.multivariate_normal(esp, cov).T
    sim = fusion_2(observation_indexes, unknown_indexes, depth, x)
    res = sim[1]
    length = [longueur(sim[1], delta)]
    for i in range(num-1):
        x = np.random.multivariate_normal(esp, cov).T
        sim = fusion_2(observation_indexes, unknown_indexes, depth, x)
        res = np.vstack((res, sim[1]))
        length.append(longueur(sim[1], delta))
    # return np.mean(res, axis=0), length/num
    return res, np.array(length)


# Simulation 100
res = simulation(esp_con, cov_cdt, Delta, 100)
# print(res[0].shape, res[1].shape)
plt.plot(unknown_distance, esp_con)
plt.plot(discretization, np.mean(res[0], axis=0))
plt.title('100 simulations et espérance condiionnelle')
plt.show()

# Question 10 # Question 11
# En fonction du nombre de simulations
# nr_sim = np.array([2**i for i in range(12)[1:]])
# re_sim = [simulation(esp_con, cov_cdt, Delta, nr_sim[i])[1] for i in range(11)]
# nr_sim = np.log2(nr_sim)
nr_sim = np.arange(100)
re_sim = simulation(esp_con, cov_cdt, Delta, 100)
plt.bar(nr_sim, re_sim[1])
plt.title('Distances des 100 simulation')
plt.show()

# Question 12

# Question 13
selection = [x for x in re_sim[1] if x >= 525]
# print(selection)
p = 100*len(selection)/N
print(f"La probabilité que la longueur du câble dépasse 525 m = {p:.2f}%")


# Question 14
# num_sim = 1000
