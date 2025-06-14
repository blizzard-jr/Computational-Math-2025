import math
import random
import numpy as np

def generate_polynomial(degree, min_val, max_val):
    coeffs = [random.randint(min_val, max_val) for _ in range(degree + 1)]
    def poly(x_val):
        return sum(coeffs[i] * x_val**i for i in range(len(coeffs)))
    return poly, coeffs

def derive_polynomial(coeffs):
    derived = [i * coeffs[i] for i in range(1, len(coeffs))]
    def derived_poly(x_val):
        return sum(derived[i] * x_val**i for i in range(len(derived)))
    return derived_poly, derived

def sample_func(x):
    return x**4 / 10 + x**2 / 5 - 7

def first_derivative(x):
    return 0.4 * x**3 + 0.4 * x

def second_derivative(x):
    return 1.2 * x**2 + 0.4

def third_derivative(x):
    return 2.4 * x

def fourth_derivative(x):
    return 2.4

def rectangle_integrate_full(f, f_dd, left, right, eps, variant, parts=100):
    grid = np.linspace(left, right, parts)
    max_d2 = max(abs(f_dd(pt)) for pt in grid)
    n = int(((max_d2 * (right - left) ** 3) / (24 * eps)) ** 0.5) + 1
    prev = rectangle_integrate(f, left, right, n, variant)
    for iteration in range(1000):
        n *= 2
        curr = rectangle_integrate(f, left, right, n, variant)
        if abs(curr - prev) < eps:
            return curr, iteration + 1
        prev = curr
    return prev, 1000

def rectangle_integrate(f, left, right, segments, variant):
    h = (right - left) / segments
    acc = 0
    if variant == 1:
        acc = sum(f(left + i * h) for i in range(segments))
    elif variant == 2:
        acc = sum(f(left + i * h) for i in range(1, segments + 1))
    else:
        acc = sum(f(left + (i + 0.5) * h) for i in range(segments))
    return acc * h

def trapezoid_integrate_full(f, f_dd, left, right, eps, steps=100):
    points = np.linspace(left, right, steps)
    m2 = max(abs(f_dd(x)) for x in points)
    n = int(((m2 * (right - left)**3) / (12 * eps))**0.5) + 2
    prev = trapezoid_integrate(f, left, right, n)
    for iteration in range(1000):
        n *= 2
        curr = trapezoid_integrate(f, left, right, n)
        if abs(curr - prev) <= eps:
            return curr, iteration + 1
        prev = curr
    return prev, 1000

def trapezoid_integrate(f, a, b, n):
    h = (b - a) / n
    acc = 0.5 * (f(a) + f(b))
    acc += sum(f(a + i * h) for i in range(1, n))
    return acc * h

def simpsons_integrate_full(f, d4f, a, b, eps, precision=100):
    samples = np.linspace(a, b, precision)
    m4 = max(abs(d4f(x)) for x in samples)
    n = int(((m4 * (b - a) ** 5) / (180 * eps)) ** 0.25)
    if n % 2 != 0:
        n += 1
    prev = simpsons_integrate(f, a, b, n)
    for iteration in range(1000):
        n *= 2
        curr = simpsons_integrate(f, a, b, n)
        if abs(curr - prev) <= eps:
            return curr, iteration + 1
        prev = curr
    return prev, 1000

def simpsons_integrate(f, a, b, n):
    h = (b - a) / n
    acc = f(a) + f(b)
    for i in range(1, n):
        coeff = 4 if i % 2 else 2
        acc += coeff * f(a + i * h)
    return acc * h / 3

def improper_integral(f, f_dd, a, b, eps, singular_point, middle=0):
    delta = 0.001 * (b - a)
    result = None
    if singular_point == 'a':
        for _ in range(500):
            current, _ = trapezoid_integrate_full(f, f_dd, a + delta, b, eps)
            if result is not None and abs(current - result) < eps:
                return current
            result = current
            delta /= 2
    elif singular_point == 'b':
        for _ in range(500):
            current, _ = trapezoid_integrate_full(f, f_dd, a, b - delta, eps)
            if result is not None and abs(current - result) < eps:
                return current
            result = current
            delta /= 2
    elif singular_point == 'c':
        for _ in range(500):
            left, _ = trapezoid_integrate_full(f, f_dd, a, middle - delta, eps)
            right, _ = trapezoid_integrate_full(f, f_dd, middle + delta, b, eps)
            total = left + right
            if result is not None and abs(total - result) < eps:
                return total
            result = total
            delta /= 2
