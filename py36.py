import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
import sympy as sp


def generate_points(func, x_range, num_points=30):
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = func(x)
    return x, y


def polynomial_fit_and_plot(x, y, max_degree=10):
    plt.figure(figsize=(14, 10))

    r_squared_values = []
    degrees = list(range(1, max_degree + 1))

    for degree in degrees:
        coefs = np.polyfit(x, y, degree)
        poly = np.poly1d(coefs)

        y_fit = poly(x)

        # R-squared 계산
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        r_squared_values.append(r_squared)

        # 다항식의 식 출력
        poly_str = " + ".join([f"{coef:.4f}*x^{i}" for i, coef in enumerate(coefs[::-1])])
        print(f'Degree {degree} Polynomial Fit: {poly_str}')

        plt.subplot(5, 2, degree)
        plt.scatter(x, y, color='blue', label='Original Points')
        plt.plot(x, y_fit, color='red', label=f'Poly Fit (Degree {degree})')
        plt.title(f'Polynomial Fit of Degree {degree} (R²: {r_squared:.4f})')
        plt.legend()
        plt.tight_layout()

        print(f'Degree {degree} Polynomial Fit R²: {r_squared:.4f}')

    plt.figure(figsize=(8, 6))
    plt.plot(degrees, r_squared_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('R-squared')
    plt.title('Polynomial Degree vs. R-squared')
    plt.grid(True)
    plt.show()


# 사용자 입력 받기
expr_input = input("함수를 입력하세요 (예: sin(x), x**2 + 3*x + 2): ")
x_min = float(input("x의 최소값을 입력하세요: "))
x_max = float(input("x의 최대값을 입력하세요: "))

# 입력된 식을 함수로 변환
x_symbol = sp.symbols('x')
func_expr = sp.sympify(expr_input)
func = sp.lambdify(x_symbol, func_expr, modules=['numpy'])

# 점 생성 및 다항식 피팅 및 플로팅
x_range = (x_min, x_max)
x, y = generate_points(func, x_range)
polynomial_fit_and_plot(x, y)