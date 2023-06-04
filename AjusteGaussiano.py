import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Datos de ejemplo
x = np.array([6, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8])
y = np.array([949.2, 944.8, 1007.8, 1223.4, 1519.2, 1569.2, 1462.6, 1097.4, 870.4])

# Definir la función de ajuste gaussiano
def gaussiana(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))

# Realizar el ajuste gaussiano
popt, pcov = curve_fit(gaussiana, x, y)

# Parámetros del ajuste
a_fit, b_fit, c_fit = popt

# Imprimir los parámetros del ajuste
print("Parámetros del ajuste:")
print("a =", a_fit)
print("b =", b_fit)
print("c =", c_fit)

# Graficar los datos y el ajuste
plt.scatter(x, y, label='Datos')
plt.plot(x, gaussiana(x, a_fit, b_fit, c_fit), 'r-', label='Ajuste')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()