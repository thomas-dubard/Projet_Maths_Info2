import numpy as np
import matplotlib.pyplot as plt
import math

t0 = 0.  # permet de changer facilement la condition initiale en temps
x0 = 1.
eps = 1e-3
n = 50  # permet de changer facilement le nombre d'itérations que l'on va effectuer sur toutes les méthodes


# Euler explicite d'ordre 1
def solve_euler_explicit(f, x0, dt=0.01, t0=0, tf=1):
    t = [t0]  # on va prendre la convention que la condition initiale x0 est en t0 qui est global sur tout le programme
    x = [x0]
    while t[-1] < tf and x[-1] < 100:
        x.append(x[-1] + dt * f(t[-1], x[-1]))
        t.append(t[-1] + dt)
    t = np.array(t[:-1])
    x = np.array(x[:-1])
    return t, x


# Euler explicite d'ordre 1
def solve_euler_explicit_2(f, x0, dt=0.01, t0=0, tf=1):
    t = [t0]  # on va prendre la convention que la condition initiale x0 est en t0 qui est global sur tout le programme
    x = [x0]
    while t[-1] < tf and x[-1] < 100:
        x.append(x[-1] + dt * f(t[-1], x[-1]))
        t.append(t[-1] + dt)
    t = np.array(t[:-1])
    x = np.array(x[:-1])
    return t, x


# Cette fonction calcule l'esperance d'une liste de valeurs pour permettre plus tard la realisation d'une regression lineaire.
def esperance(X):
    s = 0
    N = len(X)
    for x in X:
        s += x
    return s / N


# Cette fonction calcule la variance d'une liste de valeurs pour permettre plus tard la realisation d'une regression lineaire.
def variance(X):
    X2 = [x**2 for x in X]
    return esperance(X2) - (esperance(X)**2)


# Cette fonction calcule la covariance de deux listes de valeurs de meme taille pour permettre plus tard la realisation d'une regression lineaire
def covariance(X, Y):
    X_Y = [X[k]*Y[k] for k in range(len(X))]
    return esperance(X_Y) - esperance(X)*esperance(Y)


# Cette fonction realise la regression lineaire d'une nuage de points dont les abscisses sont les valeurs de la liste X et les ordonnees celles de la liste Y. Elle renvoie la pente a et l'ordonnee a l'origine b.
def regression_lineaire(X, Y):
    e_x = esperance(X)
    e_y = esperance(Y)
    v_x = variance(X)
    cov_xy = covariance(X, Y)
    a = cov_xy / v_x
    b = - (a * e_x) + e_y
    return a, b


# il faut ensuite tester cette fonction sur une équa diff facile à résoudre et montrer la convergence du schéma à l'ordre 1
def f_test(t, x):
    return math.exp(x)


dt = [3*1e-2, 1e-2, 1e-3, 1e-4]
for i in range(len(dt)):
    t_test, x_test = solve_euler_explicit(f_test, x0, dt[i], t0, t0 + 1)
    x_real = np.exp(t_test)
    plt.subplot(100*len(dt)+i+11)
    plt.plot(t_test, x_test, color='red', label='Values simulated')
    plt.plot(t_test, x_real, color='blue', label='Values calculated')
plt.show()

# Calcul approximatif de la convergence à partir d'éléments issus de mon TIPE

# ensuite, on calcule l'erreur sur le dernier terme/l'aire et on fait une régression linéaire en log

# puis sur la meme courbe on ajoute le résultat qu'on aurait eu pour une methode d'ordre 2
