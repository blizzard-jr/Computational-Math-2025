import math
import random
import numpy as np
import matplotlib.pyplot as plt

def r_uravnenie(degree, min_k, max_k):

    koefs = [random.randint(min_k, max_k) for _ in range(degree + 1)]
    def uravnenie(x):
        return sum(koefs[i] * x**i for i in range(len(koefs)))
    return uravnenie, koefs


def plot_function_system(f_system, x_min, x_max, y_min, y_max, root, title="График системы с решением"):
    def f1(x, y):
        return f_system([x, y])[0]

    def f2(x, y):
        return f_system([x, y])[1]

    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    Z1 = np.zeros_like(X)
    Z2 = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z1[i, j] = f1(X[i, j], Y[i, j])
            Z2[i, j] = f2(X[i, j], Y[i, j])

    plt.figure(figsize=(7, 6))

    plt.contour(X, Y, Z1, levels=[0], colors='red')
    plt.contour(X, Y, Z2, levels=[0], colors='blue')


    x, y = root
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)

    plt.plot(x, y, 'go', label=f'Решение ({x:.4f}, {y:.4f})')
    plt.text(x, y, f"({x:.4f}, {y:.4f})", fontsize=9, ha='left', va='bottom')

    plt.grid(True)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(title)
    plt.legend()
    plt.show()


def p_r_uravnenie(koefs):

    p_koefs = []
    for i in range(1, len(koefs)):
        p_koefs.append(i * koefs[i])
    def p_uravnenie(x):
        return sum(p_koefs[i] * x**i for i in range(len(p_koefs)))
    return p_uravnenie, p_koefs



def f_example(x):
    # return math.cos(2*x) - 7 * math.sin(x) - 4
    return x**3 - x + 4

def df_example(x):
    # return -2 * math.sin(2 * x) - 7 * math.cos(x)
    return 3*x**2 - 1



def f_system(vars_xy):
    x, y = vars_xy
    # return np.array([
    #     math.cos(x - 1) + y - 0.5,
    #     x - math.cos(y) - 3
    # ])
    return np.array([
        x**2 + y**2 - 4,
        y - 3*x**2
    ])

def df_system(vars_xy):
    x, y = vars_xy
    # return np.array([
    #      [-math.sin(x - 1), 1],
    #     [1, math.sin(y)]
    # ])
    return np.array([
        [2*x, 2*y],
        [-6*x, 1]
    ])

def plot_function(f, a, b,root,  title="График функции"):

    x_vals = np.linspace(a, b, 400)
    y_vals = [f(x) for x in x_vals]
    plt.figure(figsize=(7, 4))
    plt.plot(x_vals, y_vals, label='f(x)')
    plt.axhline(0, color='black', linewidth=0.8)

    y_root = f(root)
    plt.plot(root, y_root, 'ro', label=f'Решение: x={root:.4f}')
    plt.text(root, y_root, f"({root:.4f}, {y_root:.4f})", fontsize=9, ha='left', va='bottom')

    plt.grid(True)
    plt.xlim(a, b)
    ymin, ymax = min(y_vals), max(y_vals)
    plt.ylim(ymin - abs(ymin)*0.1, ymax + abs(ymax)*0.1)
    plt.title(title)
    plt.legend()
    plt.show()


def bisection_method(f, a, b, accuracy, m_c):
    if f(a) * f(b) > 0:
        raise ValueError("На отрезке [a, b] функция не меняет знак. Корень может отсутствовать или быть не единственным.")

    count = 0
    while count < int(m_c):
        count += 1
        x = (a + b) / 2
        if abs(f(x)) <= accuracy or abs(b - a) <= accuracy:
            return x, count, f(x)

        if f(a) * f(x) < 0:
            b = x
        else:
            a = x

    x = (a + b) / 2
    return x, count, f(x)


def secant_method(f, x0, x1, accuracy, m_c):
    count = 0
    for _ in range(int(m_c)):
        count += 1
        value_in_x0 = f(x0)
        value_in_x1 = f(x1)
        if abs(value_in_x1 - value_in_x0) < 1e-14:
            raise ZeroDivisionError("Разница f(x1) и f(x0) слишком мала, деление невозможно.")
        x2 = x1 - value_in_x1 * (x1 - x0) / (value_in_x1 - value_in_x0)
        if abs(x2 - x1) < accuracy:
            return x2, count, f(x2)
        x0, x1 = x1, x2
    return x2, count, f(x2)


def iteration_easy_method(fi, x0, accuracy, m_c):
    count = 0
    for _ in range(int(m_c)):
        count += 1
        x1 = fi(x0)
        if abs(x1 - x0) < accuracy:
            return x1, count
        x0 = x1
    return x0, count


def compute_alpha(df, a, b, steps=100):
    mas_val = np.linspace(a, b, steps)
    max_val = max(abs(df(x)) for x in mas_val)
    if (max_val == 0):
        max_val = 1
    return 1.0 / max_val



def newton_method_system(fun, dfun, x0, accuracy, count):
    x = np.array(x0, dtype=float)
    for i in range(count):
        F = fun(x)
        dF = dfun(x)
        try:
            delta = np.linalg.solve(dF, F)
        except np.linalg.LinAlgError:
            return x, i
        x_new = x - delta


        if abs(x_new[0] - x[0]) <= accuracy and abs(x_new[1] - x[1]) <= accuracy:
            return x_new, i + 1
        x = x_new
    return x, count




def main():
    print("Хотите решить ОДНОМЕРНОЕ уравнение или СИСТЕМУ?")
    print("1) Одномерное уравнение")
    print("2) Система (метод Ньютона)")
    choice_global = input("Выберите (1 или 2): ")

    if choice_global == '1':
        print("1) Использовать случайное уравнение?")
        print("2) Использовать функцию ?")
        choice = input("Выберите (1 или 2): ")

        if choice == '1':
            deg = int(input("Введите степень случайного уравнения: "))
            poly, coeffs = r_uravnenie(deg, -5, 5)
            print(f"Случайно сгенерированное уравнение (коэффициенты): {coeffs[::-1]}")
            f = poly
            dpoly, dcoeffs = p_r_uravnenie(coeffs)
            df = dpoly
        else:
            f = f_example
            df = df_example

        print("Введите границы интервала (a b): ")
        inp = input().split()
        a = float(inp[0])
        b = float(inp[1])

        # plot_function(f, a, b, title="График выбранной функции на интервале")

        print("Введите требуемую точность: ")
        accuracy = float(input())

        print("Введите максимальное число итераций: ")
        count = int(input())

        print("Какой метод использовать?")
        print("1) Метод половинного деления")
        print("2) Метод секущих")
        print("3) Метод простой итерации")
        choice_method = input("Выберите (1, 2, 3): ")

        if choice_method == '1':
            try:
                root, n_iter, f_val = bisection_method(f, a, b, accuracy, count)
                print(f"\nМетод половинного деления:\nНайденный корень: {root}\nКоличество итераций: {n_iter}\nf(root) = {f_val}")
                plot_function(f, a, b, root, title="График выбранной функции на интервале")
            except ValueError as e:
                print(f"Ошибка: {e}")

        elif choice_method == '2':
            try:
                root, n_iter, f_val = secant_method(f, a, b, accuracy, count)
                print(f"\nМетод секущих:\nНайденный корень: {root}\nКоличество итераций: {n_iter}\nf(root) = {f_val}")
                plot_function(f, a, b, root, title="График выбранной функции на интервале")
            except ZeroDivisionError as e:
                print(f"Ошибка: {e}")

        elif choice_method == '3':
            alpha = compute_alpha(df, a, b)
            print(f"Вычислен alpha = {alpha}")
            def fi(x):
                return x - alpha * f(x)
            x0 = (a + b) / 2

            root, n_iter = iteration_easy_method(fi, x0, accuracy, count)
            print(f"\nМетод простой итерации:\nНайденный корень: {root}\nКоличество итераций: {n_iter}\nf(root) = {f(root)}")
            plot_function(f, a, b, root, title="График выбранной функции на интервале")
        else:
            print("Неправильный выбор метода!")

    else:
        print("Решаем систему методом Ньютона.")
        print("Для демонстрации возьмём систему со слайдов:\n  1) x^2 + y^2 = 4\n  2) y - 3x^2 = 0\n")

        print("Введите начальное приближение (x0, y0): ")
        inp = input().split()
        x0 = float(inp[0])
        y0 = float(inp[1])

        print("Введите требуемую точность: ")
        accuracy = float(input())

        print("Введите максимальное число итераций: ")
        count = int(input())

        root, iters = newton_method_system(f_system, df_system, [x0, y0], accuracy, count)
        if (root[0] == x0 and root[1] == y0) and (f_system([x0, y0])[0] != 0 or f_system([x0, y0])[1] != 0):
            root, iters = newton_method_system(f_system, df_system, [x0 + accuracy, y0 + accuracy], accuracy, count)
        print(f_system([0, 0]))
        print("\nМетод Ньютона")
        print(f"Решения: x = {root[0]:.6f}, y = {root[1]:.6f}")
        print(f"Число итераций: {iters}")
        print(f"f1(x,y) = {f_system(root)[0]:.6e}")
        print(f"f2(x,y) = {f_system(root)[1]:.6e}")
        plot_function_system(f_system, root[0] - 3, root[0] + 3, root[1] - 3, root[1]  + 3, root, title="График системы с решением")


if __name__ == "__main__":
    main()
