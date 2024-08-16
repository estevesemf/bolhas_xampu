import numpy as np
import matplotlib.pyplot as plt
import math

def euler(f,y0,x0,h,precisao):
    '''Solução para y'=f(y,x) usando o método de Euler (RK1).
    ----------
    f: função, y'=f(y,x)
    y0: valor inicial de y, y(x0)
    x0: valor inicial de x
    h: distância entre os x_n
    precisao: precisão
    ----------
    Retorna o conjunto de pontos y(x)
    '''

    y = []
    y.append(y0)
    x = []
    x.append(x0)

    n = 0
    while(True):
      x.append(x[n] + h)
      y.append(y[n] + h*f(y[n],x[n]))
      n += 1

      if((y[n]-y[n-1])/y[n] < precisao):
        break

    return y, x

def euler_aperfeicoado(f,y0,x0,h,precisao):
    '''Solução para y'=f(y,x) usando o método de Euler Melhorado (RK2).
    ----------
    f: função, y'=f(y,x)
    y0: valor inicial de y, y(x0)
    x0: valor inicial de x
    h: distância entre os x_n
    precisao: precisão
    ----------
    Retorna o conjunto de pontos y(x)
    '''

    y = []
    y.append(y0)
    x = []
    x.append(x0)

    n = 0
    while(True):
      x.append(x[n] + h)
      y.append(y[n] + 0.5*h*(f(y[n],x[n]) + f(y[n]+h*f(y[n],x[n]),x[n]+h)))
      n += 1

      if((y[n]-y[n-1])/y[n] < precisao):
        break

    return y, x

def rk4(f,y0,x0,h,precisao):
    '''Solução para y'=f(y,x) usando Runge-Kutta de ordem 4.
    ----------
    f: função, y'=f(y,x)
    y0: valor inicial de y, y(x0)
    x0: valor inicial de x
    h: distância entre os x_n
    precisao: precisão
    ----------
    Retorna o conjunto de pontos y(x)
    '''

    y = []
    y.append(y0)
    x = []
    x.append(x0)

    n = 0
    while(True):
      x.append(x[n] + h)

      k1 = h*f(y[n], x[n])
      k2 = h*f(y[n]+k1/2, x[n]+h/2)
      k3 = h*f(y[n]+k2/2, x[n]+h/2)
      k4 = h*f(y[n]+k3, x[n]+h)

      y.append(y[n] + (k1 + 2*k2 + 2*k3 + k4)/6)
      n += 1

      if((y[n]-y[n-1])/y[n] < precisao):
        break

    return y, x

R = 2.5
densAr = 1.25E-6
densX = 1.03E-3
g = 9.78E3
n = 1.7
b = 420

f = lambda v,t: (((densX-densAr)*g/densAr)-3*b*(v**n)/(4*math.pi*R*R*R*densAr))
v0 = 0
t0 = 0
h = 1E-9
precisao = 1E-5

v_rk1,t_rk1 = euler(f,v0,t0,h,precisao)
v_rk2,t_rk2 = euler_aperfeicoado(f,v0,t0,h,precisao)
v_rk4,t_rk4 = rk4(f,v0,t0,h,precisao)

print(f"Velocidade limite usando RK1: {v_rk1[-1]}mm/s")
print(f"Velocidade limite usando RK2: {v_rk2[-1]}mm/s")
print(f"Velocidade limite usando RK4: {v_rk4[-1]}mm/s")
print(f"Precisão: {precisao}mm/s")

print(f"Tempo até atingir a velocidade limite (RK1): {round(t_rk1[-1]*1E9,4)}ns")
print(f"Tempo até atingir a velocidade limite (RK2): {round(t_rk2[-1]*1E9,4)}ns")
print(f"Tempo até atingir a velocidade limite (RK4): {round(t_rk4[-1]*1E9,4)}ns")

plt.title(f"Bolha de ar em xampu (usando Euler):")
plt.grid(True)
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade (mm/s)')
plt.plot(t_rk1,v_rk1,'b')
plt.legend(['rk1'])
plt.show()

plt.title(f"Bolha de ar em xampu (usando Euler Aperfeiçoado):")
plt.grid(True)
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade (mm/s)')
plt.plot(t_rk2,v_rk2,'g')
plt.legend(['rk2'])
plt.show()

plt.title(f"Bolha de ar em xampu (usando RK4):")
plt.grid(True)
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade (mm/s)')
plt.plot(t_rk4,v_rk4,'r')
plt.legend(['rk4'])
plt.show()

plt.title(f"Comparação:")
plt.grid(True)
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade (mm/s)')
plt.plot(t_rk4,v_rk4,'r', t_rk2,v_rk2,'g', t_rk1,v_rk1,'b')
plt.legend(['rk4', 'rk2', 'rk1'])
plt.show()
