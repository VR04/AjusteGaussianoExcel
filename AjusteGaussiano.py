import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Datos de ejemplo
X = np.array([ 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, ])
Y = np.array([ 1007.8, 1223.4, 1519.2, 1569.2, 1462.6, 1097.4, 870.4 ])

# Función de ajuste gaussiana
def gaussian_func(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))

# Realizar el ajuste con valores iniciales y más iteraciones
params, cov = curve_fit(gaussian_func, X, Y, p0=[1000, 6, 1], maxfev=10000)

# Parámetros del ajuste
a_fit, b_fit, c_fit = params

# Calcular el coeficiente de ajuste (R cuadrado)
y_fit = gaussian_func(X, a_fit, b_fit, c_fit)
r_squared = np.corrcoef(Y, y_fit)[0, 1]**2

# Rango para la curva ajustada
x_fit = np.linspace(X.min(), X.max(), 100)
y_fit = gaussian_func(x_fit, a_fit, b_fit, c_fit)

# Graficar los datos y el ajuste
plt.scatter(X, Y, label='Datos')
plt.plot(x_fit, y_fit, 'r-', label='Ajuste')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Imprimir los parámetros del ajuste y el coeficiente de ajuste
print("Parámetros del ajuste:")
print("a =", a_fit)
print("b =", b_fit)
print("c =", c_fit)
print("Coeficiente de ajuste (R cuadrado) =", r_squared)
