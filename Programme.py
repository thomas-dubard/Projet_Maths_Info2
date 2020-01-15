import numpy as np
import matplotlib.pyplot as plt
import math
from math import log

t0 = 0.  # Permet de changer facilement la condition initiale en temps
x0 = 1.


def f(x):
    return x
# Ici on utilise la fonction exponentielle pour tester les méthodes
# A priori il faut définir comme f(t, x)
# Mais dans votre fonction donnée pour Euler explicie à pas variable
# Vous mettez seulement une variable pour la fonction f dans ses arguments
# Donc on a enlevé la variable t en cosidérant que f est indépendante du temps


# Euler explicite d'ordre 1
def solve_euler_explicit(f, x0, dt=0.01, t0=0, tf=1):
    t = [t0]
    # On va prendre la convention que la condition initiale x0 est en t0
    x = [x0]
    while t[-1] < tf:
        x.append(x[-1] + dt * f(x[-1]))
        t.append(t[-1] + dt)
    t = np.array(t[:-1])
    x = np.array(x[:-1])
    return t, x


# Euler explicite d'ordre 2: méthode de Heun
def solve_heun_explicit(f, x0, dt=0.01, t0=0, tf=1):
    t = [t0]
    # on va prendre la convention que la condition initiale x0 est en t0 qui
    # est global sur tout le programme
    x = [x0]
    while t[-1] < tf:
        swap = f(t[-1], x[-1]) + f(t[-1], x[-1] + dt * f(t[-1], x[-1]))
        x.append(x[-1] + (dt/2) * swap)
        t.append(t[-1] + dt)
    t = np.array(t[:-1])
    x = np.array(x[:-1])
    return t, x


dt = [3*1e-2, 1e-2, 1e-3, 1e-4]
fig, axes = plt.subplots(2, 2, figsize=(20, 20))
for i in range(len(dt)):
    t_test, x_test = solve_euler_explicit(f, x0, dt[i], t0, t0 + 40)
    x_real = np.exp(t_test)
    axes[i % 2][i//2].plot(t_test[1000:], x_test[1000:],
                           color='red', label='Values simulated')
    axes[i % 2][i//2].plot(t_test[1000:], x_real[1000:],
                           color='blue', label='Values calculated')
    axes[i % 2][i//2].set_title(f"Euler explicite depuis 0 pour dt={dt[i]}")
plt.show()

# dt = [3*1e-2, 1e-2, 1e-3, 1e-4]
# for i in range(len(dt)):
#     t_test, x_test = solve_euler_explicit(f_test, x0, dt[i], t0, t0 + 1)
#     x_real = np.exp(t_test)
#     ax = plt.subplot(100*int(len(dt)/2)+(i)*1+21)
#     ax.set_title(f'Delta t ={dt[i]}')
#     plt.plot(t_test, x_test, color='red', label='Values simulated')
#     plt.plot(t_test, x_real, color='blue', label='Values calculated')
# plt.show()

# Attention si le programme ne s'éxécute pas assez vite
# réduire ttf quitte à perdre de l'info
# En effet, la valeur de ttf acceptable dépend des performances du CPU
# Notre ordinateur a planté pour ttf=70...
# En pratique on voudrait une grande valeur
# mais problème de mémoire dès ttf=10, et ça rame même à ttf=2
ttf = t0 + 2
dt = [1e-1, 1e-2, 1e-3, 1e-4]
err_euler = []
maj = []
cste = 10
for i in range(len(dt)):
    t_test, x_test = solve_euler_explicit(f, x0, dt[i], t0, ttf)
    x_real = np.exp(t_test)
    err_euler.append(max(np.abs(x_test-x_real)))
    maj.append(dt[i]*cste)
plt.plot([-log(x, 10) for x in dt], err_euler,
         color='red', label='Values simulated')
plt.plot([-log(x, 10) for x in dt], maj, color='blue', label='Majoration')
plt.show()


def solve_heun_explicit(f, x0, dt=0.01, t0=0, tf=1):
    t = [t0]
    # On va prendre la convention que la condition initiale x0 est en t0
    x = [x0]
    while t[-1] < tf:
        swap = f(x[-1]) + f(x[-1] + dt * f(x[-1]))
        x.append(x[-1] + (dt/2) * swap)
        t.append(t[-1] + dt)
    t = np.array(t[:-1])
    x = np.array(x[:-1])
    return t, x


dt = [3*1e-2, 1e-2, 1e-3, 1e-4]
fig, axes = plt.subplots(2, 2, figsize=(20, 20))
for i in range(len(dt)):
    t_test, x_test = solve_heun_explicit(f, x0, dt[i], t0, t0 + 40)
    x_real = np.exp(t_test)
    axes[i % 2][i//2].plot(t_test[1000:], x_test[1000:],
                           color='red', label='Values simulated')
    axes[i % 2][i//2].plot(t_test[1000:], x_real[1000:],
                           color='blue', label='Values calculated')
    axes[i % 2][i//2].set_title(f"Heun explicite depuis 0 pour dt={dt[i]}")
plt.show()

dt = [1e-1, 1e-2, 1e-3, 1e-4]
err_heun = []
maj_heun = []
cste = 10
for i in range(len(dt)):
    t_test, x_test = solve_heun_explicit(f, x0, dt[i], t0, ttf)
    x_real = np.exp(t_test)
    err_heun.append(max(np.abs(x_test-x_real)))
    maj_heun.append((dt[i])**2*cste)
plt.plot([-log(x, 10) for x in dt], err_heun,
         color='red', label='Values simulated')
plt.plot([-log(x, 10) for x in dt], maj_heun, color='blue', label='Majoration')
plt.show()

plt.plot([-log(x, 10) for x in dt], err_heun, color='blue',
         label='Erreurs de méthode d\'ordre 2')
plt.plot([-log(x, 10) for x in dt], err_euler, color='red',
         label='Erreurs de méthode d\'ordre 1')
plt.show()


def solve_ivp_euler_explicit_variable_step(f, t0, x0, t_f, dtmin=1e-16,
                                           dtmax=0.01, atol=1e-6):
    dt = dtmax/10  # Initial integration step
    ts, xs = [t0], [x0]  # Storage variables
    t = t0
    ti = 0  # Internal time keeping track of time since latest storage point
    x = x0
    while ts[-1] < t_f:
        while ti < dtmax:  # must remain below dtmax
            t_next, ti_next, x_next = t + dt, ti + dt, x + dt * f(x)
            x_back = x_next - dt * f(x_next)
            ratio_abs_error = atol / (linalg.norm(x_back-x)/2)
            dt = 0.9 * dt * sqrt(ratio_abs_error)
            if dt < dtmin:
                raise ValueError("Time step below minimum")
            elif dt > dtmax/2:
                dt = dtmax/2
            t, ti, x = t_next, ti_next, x_next
        dt2DT = dtmax - ti  # Time left to dtmax
        t_next, ti_next, x_next = t + dt2DT, 0, x + dt2DT * f(x)
        ts = vstack([ts, t_next])
        xs = vstack([xs, x_next])
        t, ti, x = t_next, ti_next, x_next
    return (ts, xs.T)


# %matplotlib inline
dt = [3*1e-2, 1e-2, 1e-3, 1e-4]
fig, axes = plt.subplots(2, 2, figsize=(20, 20))
for i in range(len(dt)):
    (t_test, x_test) = solve_ivp_euler_explicit_variable_step(
        f, t0, x0, t0 + 2, dtmax=dt[i])
    t_test = t_test.T[0]
    x_test = x_test[0]
    x_real = np.exp(t_test)
    axes[i % 2][i//2].plot(t_test, x_test, color='red',
                           label='Values simulated')
    axes[i % 2][i//2].plot(t_test, x_real, color='blue',
                           label='Values calculated')
    axes[i %
         2][i//2].set_title(f"Euler explicite à pas variable pour dt={dt[i]}")
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(20, 20))
for i in range(len(dt)):
    (t_test, x_test) = solve_ivp_euler_explicit_variable_step(
        f, t0, x0, t0 + 2, dtmax=dt[i])
    t_test = t_test.T[0]
    x_test = x_test[0]
    t_euler, x_euler = solve_euler_explicit(f, x0, dt[i], t0, t0 + 2)
    x_real = np.exp(t_test)
    axes[i % 2][i//2].plot(t_test, x_real, color='green',
                           label='Valeur réelle')
    axes[i % 2][i//2].plot(t_test, x_test, color='red',
                           label='Euleur à pas variable')
    axes[i % 2][i//2].plot(t_euler, x_euler, color='blue',
                           label='Euleur à pas fixe')
    axes[i %
         2][i//2].set_title(f"Euler explicite à pas variable pour dt={dt[i]}")
plt.show()

dt = [1e-1, 1e-2, 1e-3, 1e-4]
err_euler = []
for i in range(len(dt)):
    (t_test, x_test) = solve_ivp_euler_explicit_variable_step(
        f, t0, x0, ttf, dtmax=dt[i])
    t_test = t_test.T[0]
    x_test = x_test[0]
    x_real = np.exp(t_test)
    err_euler.append(max(np.abs(x_test-x_real)))
plt.plot([-log(x, 10) for x in dt], err_euler,
         color='red', label='Erreurs de simulation')
plt.show()

err_euler_fixe = []
for i in range(len(dt)):
    t_test, x_test = solve_euler_explicit(f, x0, dt[i], t0, ttf)
    x_real = np.exp(t_test)
    err_euler_fixe.append(max(np.abs(x_test-x_real)))
    maj.append(dt[i]*cste)
plt.plot([-log(x, 10) for x in dt], err_euler, color='red',
         label='Erreurs d\'Euleur à pas variable')
plt.plot([-log(x, 10) for x in dt], err_euler_fixe,
         color='blue', label='Erreurs d\'Euleur à pas fixe')
plt.show()
