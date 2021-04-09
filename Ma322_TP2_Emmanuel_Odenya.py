######################## Ma322 TP2 -- Emmanuel ODENYA #####################


### IMPORTATIONS

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


############### Oscillations du pendule #################

### FONCTIONS
''' Dans cette partie sont recensés tous les programmes et fonctions
    écrites pour ce TP(cas du pendule). Les représentations graphiques sont
    présentées à la suite. La présentation sera pareil pour la suspension'''

# 4.1 Question 4

def theta(t):
    A = np.pi/2
    g = 9.81
    L = 1
    return A*(np.cos(np.sqrt(g/L)*t))

X = np.linspace(0, 4, 101)   #la commande linspace met permet de discrétiser l'intervalle [0,4] en 100 intervalles
n = X.size


# 4.2.2 Résolution par la méthode d'Euler

### Pendule (pour calculer Yprime à partir de Y et t)

g = 9.81
L = 1

# '''Formulation du problème à partir de l'équation exacte'''

def pendule(Y, t):
    return np.array([Y[1], -(g/L)*np.sin(Y[0])])  

# '''Formulation du problème à partir de l'équation linéarisée'''

def pendule_lin(Y, t):
    return np.array([Y[1], -(g/L)*Y[0]])

 # Ces deux formulations vont nous permettre par la suite d'étudier
 # la pertinence de nos hypothèses de travail


### Euler explicite (résolution du problème)

def EulerExplicite(f, h, y0):
    p = y0.size
    Ye = np.zeros((n, p))
    Ye[0, : ] = y0
    for i in range(n-1): #On stocke les valeurs successives de Y_(i+1) à partir du schéma d'Euler explicite 
        Ye[i+1, : ] = Ye[i , : ] + h*f(Ye[i , : ], 0) # le pendule étant indépendant de t, je pose t=0
    return Ye


# 4.2.3 Résolution par la méthode de Runge Kutta 4

def RungeKutta(f, h, y0):
    p = y0.size
    Y_rk = np.zeros((n,p))
    Y_rk[0, : ] = y0
    k_1 = np.zeros((n,p))
    k_2 = np.zeros((n,p))
    k_3 = np.zeros((n,p))
    k_4 = np.zeros((n,p))
    for i in range(n-1):
        k_1[i, : ] = f(Y_rk[i , : ], 0)
        k_2[i, : ] = f(Y_rk[i , : ] + (h/2)*k_1[i, : ], 0)
        k_3[i, : ] = f(Y_rk[i , : ] + (h/2)*k_2[i, : ], 0)
        k_4[i, : ] = f(Y_rk[i , : ] + (h/2)*k_3[i, : ], 0)
        Y_rk[i+1, : ] = Y_rk[i , : ] + (h/6)*(k_1[i, : ] + 2*k_2[i, : ] + 2*k_3[i, : ] + k_4[i, : ])
    return Y_rk

#Cette formulation nous permet de stocker les valeurs successives de k_1..k_4
# ce qui n'est pas forcément nécessaire ici. On pourrait tout à fait prendre
# des réels mis à jour à chaque itération

### COURBES
''' Ici sera réalisé le tracé des courbes de Theta pour le pendule '''

### Résolution avec solveur odeint

Y0 = np.array([(np.pi)/2, 0])

Yode = odeint(pendule, Y0, X)
Yode_lin = odeint(pendule_lin, Y0, X)


Theta = np.zeros(n)
for i in range(n):
    Theta[i] = theta(X[i])  #Je  calcule les valeurs successives de theta(t) que je stocke dans la table Theta


Euler = EulerExplicite(pendule, 0.04, Y0)
Euler_lin = EulerExplicite(pendule_lin, 0.04, Y0)


Runge_Kutta = RungeKutta(pendule, 0.04, Y0)
Runge_Kutta_lin = RungeKutta(pendule_lin, 0.04, Y0)


## Equation exacte

plt.plot(X, Theta, color='magenta', label = 'Problème linéarisé')
plt.plot(X, Euler[:,0], color='red', label = 'Euler explicite')
plt.plot(X, Runge_Kutta[:,0], color='green', label = 'Runge Kutta 4')
plt.plot(X, Yode[:,0], color='blue', label='Solveur odeint')
plt.xlabel('Temps')
plt.ylabel('Theta')
plt.title('Evolution de Theta pour le pendule')
plt.legend()
plt.grid()
plt.show()

## Equation linéarisée

plt.plot(X, Theta, color='orange', label = 'Problème linéarisé')
plt.plot(X, Euler_lin[:,0], color='red', label = 'Euler explicite')
plt.plot(X, Runge_Kutta_lin[:,0], color='green', label = 'Runge Kutta 4')
plt.plot(X, Yode_lin[:,0], color='blue', label='Solveur odeint')
plt.xlabel('Temps')
plt.ylabel('Theta')
plt.title('Evolution de Theta pour le pendule linéarisé')
plt.legend()
plt.grid()
plt.show()


## Test avec theta(0) = pi/12

def theta_t(t):
    A = np.pi/12
    g = 9.81
    L = 1
    return A*(np.cos(np.sqrt(g/L)*t))

Theta_t = np.zeros(n)
for i in range(n):
    Theta_t[i] = theta_t(X[i])

Y0p = np.array([(np.pi)/12, 0])

Euler_t = EulerExplicite(pendule, 0.04, Y0p)
Runge_Kutta_t = RungeKutta(pendule, 0.04, Y0p)
Yode_t = odeint(pendule, Y0p, X)


plt.plot(X, Theta_t, color='orange', label = 'Problème linéarisé')
plt.plot(X, Euler_t[:,0], color='red', label = 'Euler explicite corrigé')
plt.plot(X, Runge_Kutta_t[:,0], color='green', label = 'Runge Kutta 4 corrigé')
plt.plot(X, Yode_t[:,0], color='blue', label='Solveur odeint')
plt.xlabel('Temps')
plt.ylabel('Theta')
plt.title('Avec petits angles respectés')
plt.legend()
plt.grid()
plt.show()





### 5. Questions bonus - Portrait de phase et Runge Kutta 2

# Portrait de phase

plt.plot(Euler[:,0], Euler[:,1], label = 'Avec Euler explicite', color = 'blue')
plt.plot(Runge_Kutta[:,0], Runge_Kutta[:,1], label = 'Avec Runge Kutta 4', color ='green')
plt.plot(Yode[:,0], Yode[:,1], label = 'Avec odeint', color = 'red')
plt.xlabel('Theta')
plt.ylabel('Theta prime')
plt.title('Portrait de phase')
plt.grid()
plt.legend()
plt.show()

# Runge Kutta d'ordre 2

def RK2(f, h, y0):
    p = 2
    Y_r = np.zeros((n,p))
    Y_r[0, : ] = y0
    k_1 = np.zeros((n,p))
    k_2 = np.zeros((n,p))
    for i in range(n-1):
        k_1[i, : ] = f(Y_r[i , : ], 0)
        k_2[i, : ] = f(Y_r[i , : ] + (h/2)*k_1[i, : ], 0)
        Y_r[i+1, : ] = Y_r[i , : ] + h*k_2[i, : ]
    return Y_r

Runge = RK2(pendule, 0.04, Y0)

plt.plot(X, Runge[:,0], label = 'Runge Kutta 2')
plt.plot(X, Runge_Kutta[:,0], color='green', label = 'Runge Kutta 4')
plt.legend()
plt.title('Résolution du pendule avec Runge Kutta 2')
plt.grid()
plt.show()



# ############## Suspension d'un véhicule ##########  
    

def suspension(Y, t):
    M_1 = 15
    M_2 = 200
    C_2 = 1200
    K_1 = 50000
    K_2 = 5000
    f = -1000
    X_1p = Y[2]
    X_2p = Y[3]
    X_3p = (1/M_1)*(K_2*Y[1] - (K_1 + K_2)*Y[0] + C_2*Y[3] - C_2*Y[2])
    X_4p = (1/M_2)*( - K_2*Y[1] + K_2*Y[0] - C_2*Y[3] + C_2*Y[2] + f)
    return np.array([X_1p, X_2p, X_3p, X_4p])

Y_0 = np.array([0, 0, 0, 0])

T = np.arange(0, 3, 0.03)

Suspension = odeint(suspension, Y_0, T)
Tot = Suspension[:,0] + Suspension[:,1]

plt.plot(T, Suspension[:,0], label = '$x_1$ : roue', color = 'blue')
plt.plot(T, Suspension[:,1], label = '$x_2$ : caisse', color = 'green')
plt.plot(T, Tot, label = 'Affaissement global', color = 'red')
plt.title('Suspension du véhicule')
plt.legend()
plt.grid()
plt.show()


def suspension_v2(Y, t):
    M_1 = 15
    M_2 = 200
    C_2 = 1200
    K_1 = 50000
    K_2 = 5000
    f = -250
    X_1p = Y[2]
    X_2p = Y[3]
    X_3p = (1/M_1)*(K_2*Y[1] - (K_1 + K_2)*Y[0] + C_2*Y[3] - C_2*Y[2])
    X_4p = (1/M_2)*( - K_2*Y[1] + K_2*Y[0] - C_2*Y[3] + C_2*Y[2] + f)
    return np.array([X_1p, X_2p, X_3p, X_4p])


Suspension_v2 = odeint(suspension_v2, Y_0, T)
Tot2 = Suspension_v2[:,0] + Suspension_v2[:,1]

plt.plot(T, Suspension_v2[:,0], label = '$x_1$ : roue')
plt.plot(T, Suspension_v2[:,1], label = '$x_2$ : caisse')
plt.plot(T, Tot2, label = 'Affaissement global')
plt.title('Suspension du véhicule (ajustée)')
plt.legend()
plt.grid()
plt.show()



###### Et si f(t) dépendait du temps ??? ######

# Pour ce test, j'utilise, avec son accord la fonction "suspension" de Charles 
#(à laquelle j'apporte quelques modifications) car contrairement à la mienne
# la fonction f(t) y est décrite comme une fonction dépendant du temps.
# Par ce test, j'essaie de visualiser l'évolution de l'affaissement du 
# véhicule si la charge était une exponentielle décroissante du temps 


def f(t):      
    f_t=-250*np.exp(-t)
    return f_t

def suspension(Y,t):
    Yprime=np.zeros((4,))
    x1,x2,x3,x4=Y[0],Y[1],Y[2],Y[3]
    Yprime[0]=x3
    Yprime[1]=x4
    Yprime[2]=-(C2/M1)*x3+(C2/M1)*x4-((K1+K2)/M1)*x1+(K2/M1)*x2
    Yprime[3]=(C2/M2)*x3-(C2/M2)*x4+(K2/M2)*x1-(K2/M2)*x2+ f(t)/M2
    Yprime.reshape((4,))
    return Yprime


M1=15
M2=200
C2=1200
K1=50000
K2=5000

h = 0.03
tfin = 10 # je choisis une fin plus lointaine pour visualiser la convergence

t = np.arange(0, tfin, h)

Y0_temps = np.zeros((4,))

susp_ode = odeint(suspension,Y0_temps,t)
S = susp_ode[:,0] + susp_ode[:,1]

plt.plot(t, susp_ode[:,0], label = 'x1(t)')
plt.plot(t, susp_ode[:,1], label = 'x2(t)')
plt.plot(t, S, label = 'Ensemble',color='g')
plt.title("Suspension pour une charge qui dépend du temps")
plt.xlabel("Temps (s)")
plt.ylabel("Déplacement (m)")
plt.legend()
plt.grid()
plt.show()

# Le tracé nous montre une convergence vers 0, ce qui est cohérent avec notre
# hypothèse puisque l'exponentielle tend vers 0 en l'infini. Donc l'effet de
# la charge tend à s'annuler.


