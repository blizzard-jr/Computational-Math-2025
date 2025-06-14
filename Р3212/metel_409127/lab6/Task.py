import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class ODESolver:
    def __init__(self):
        self.equations = {
            1: {
                'name': "y' = x + y",
                'func': lambda x, y: x + y,
                'exact': lambda x, x0, y0: (y0 + x0 + 1) * np.exp(x - x0) - x - 1,
                'description': "Точное решение: y = (y₀ + x₀ + 1) * e^(x-x₀) - x - 1"
            },
            2: {
                'name': "y' = x² - 2y",
                'func': lambda x, y: x**2 - 2*y,
                'exact': lambda x, x0, y0: (y0 - x0**2/2 + x0/2 - 1/4) * np.exp(-2*(x - x0)) + x**2/2 - x/2 + 1/4,
                'description': "Точное решение: y = (y₀ - x₀²/2 + x₀/2 - 1/4) * e^(-2(x-x₀)) + x²/2 - x/2 + 1/4"
            },
            3: {
                'name': "y' = sin(x) + y",
                'func': lambda x, y: np.sin(x) + y,
                'exact': lambda x, x0, y0: (y0 + np.cos(x0)/2 + np.sin(x0)/2) * np.exp(x - x0) - np.cos(x)/2 - np.sin(x)/2,
                'description': "Точное решение: y = (y₀ + cos(x₀)/2 + sin(x₀)/2) * e^(x-x₀) - cos(x)/2 - sin(x)/2"
            }
        }
    
    def euler_method(self, f: Callable, x0: float, y0: float, xn: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
        n = int((xn - x0) / h)
        x = np.linspace(x0, xn, n + 1)
        y = np.zeros(n + 1)
        y[0] = y0
        for i in range(n):
            y[i + 1] = y[i] + h * f(x[i], y[i])
        
        return x, y
    
    def improved_euler_method(self, f: Callable, x0: float, y0: float, xn: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
        n = round((xn - x0) / (h))
        x = np.linspace(x0, xn, n + 1)
        y = np.zeros(n + 1)
        y[0] = y0
        for i in range(n):
            k1 = f(x[i], y[i])
            k2 = f(x[i] + h, y[i] + h * k1)
            y[i + 1] = y[i] + h * (k1 + k2) / 2
        
        return x, y
    
    def milne_method(self, f: Callable, x0: float, y0: float, xn: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
        n = int((xn - x0) / h)
        x = np.linspace(x0, xn, n + 1)
        y = np.zeros(n + 1)
        if n < 4:
            return self.improved_euler_method(f, x0, y0, xn, h)

        x_start, y_start = self.improved_euler_method(f, x0, y0, x0 + 3*h, h)
        y[:4] = y_start

        for i in range(4, n + 1):
            f_i_3 = f(x[i-3], y[i-3])
            f_i_2 = f(x[i-2], y[i-2])
            f_i_1 = f(x[i-1], y[i-1])
            
            y_pred = y[i-4] + (4*h/3) * (2*f_i_3 - f_i_2 + 2*f_i_1)
            
            f_i_pred = f(x[i], y_pred)
            
            y[i] = y[i-2] + (h/3) * (f_i_2 + 4*f_i_1 + f_i_pred)
        
        return x, y
    
    def runge_rule_error(self, f: Callable, x0: float, y0: float, xn: float, h: float, method: str) -> float:
        if method == 'euler':
            _, y_h = self.euler_method(f, x0, y0, xn, h)
            _, y_h2 = self.euler_method(f, x0, y0, xn, h/2)
            p = 1  # порядок
        elif method == 'improved_euler':
            _, y_h = self.improved_euler_method(f, x0, y0, xn, h)
            _, y_h2 = self.improved_euler_method(f, x0, y0, xn, h/2)
            p = 2
        
        # Сравниваем значения в одних и тех же точках
        y_h2_interp = y_h2[::2]
        
        runge_error = np.max(np.abs(y_h - y_h2_interp)) / (2**p - 1)
        return runge_error
    
    def exact_error(self, exact_func: Callable, x: np.ndarray, y_numerical: np.ndarray, x0: float, y0: float) -> float:
        y_exact = exact_func(x, x0, y0)
        return np.max(np.abs(y_exact - y_numerical))
    
    def create_results_table(self, x: np.ndarray, methods_results: dict, exact_func: Callable, x0: float, y0: float):
        print("\n" + "="*80)
        print("ТАБЛИЦА ПРИБЛИЖЕННЫХ ЗНАЧЕНИЙ")
        print("="*80)
        
        header = f"{'x':>8} "
        for method_name in methods_results.keys():
            header += f"{method_name:>15} "
        if exact_func:
            header += f"{'Точное':>15}"
        print(header)
        print("-" * len(header))
        
        step = max(1, len(x) // 10)
        for i in range(0, len(x), step):
            row = f"{x[i]:>8.3f} "
            for method_name, (_, y_vals) in methods_results.items():
                if i < len(y_vals):
                    row += f"{y_vals[i]:>15.6f} "
                else:
                    row += f"{'N/A':>15} "
            
            if exact_func:
                y_exact_val = exact_func(x[i], x0, y0)
                row += f"{y_exact_val:>15.6f}"
            
            print(row)
    
    def solve_and_analyze(self, equation_num: int, x0: float, y0: float, xn: float, h: float, epsilon: float):
        if equation_num not in self.equations:
            print("Ошибка: Неверный номер уравнения!")
            return
        
        eq = self.equations[equation_num]
        f = eq['func']
        exact_func = eq['exact']
        
        print(f"\nРешение уравнения: {eq['name']}")
        print(f"Начальные условия: y({x0}) = {y0}")
        print(f"Интервал: [{x0}, {xn}], шаг h = {h}")
        print(f"{eq['description']}")
        
        # Решение всеми методами
        x_euler, y_euler = self.euler_method(f, x0, y0, xn, h)
        x_improved, y_improved = self.improved_euler_method(f, x0, y0, xn, h)
        x_milne, y_milne = self.milne_method(f, x0, y0, xn, h)
        
        # Сохраняем результаты
        methods_results = {
            'Эйлер': (x_euler, y_euler),
            'Улучш. Эйлер': (x_improved, y_improved),
            'Милн': (x_milne, y_milne)
        }
        
        # Создаем таблицу результатов
        self.create_results_table(x_euler, methods_results, exact_func, x0, y0)
        
        # Анализ погрешностей
        print("\n" + "="*80)
        print("АНАЛИЗ ПОГРЕШНОСТЕЙ")
        print("="*80)
        
        # Правило Рунге для одношаговых методов
        euler_runge_error = self.runge_rule_error(f, x0, y0, xn, h, 'euler')
        improved_euler_runge_error = self.runge_rule_error(f, x0, y0, xn, h, 'improved_euler')
        
        print(f"Погрешность по правилу Рунге:")
        print(f"  Метод Эйлера: {euler_runge_error:.2e}")
        print(f"  Усовершенствованный метод Эйлера: {improved_euler_runge_error:.2e}")
        
        milne_exact_error = self.exact_error(exact_func, x_milne, y_milne, x0, y0)

        print(f"\nТочная погрешность (ε = max|y_точн - y_i|):")
        print(f"  Метод Милна: {milne_exact_error:.2e}")
        
        # Построение графиков
        plt.figure(figsize=(12, 8))
        
        # График численных решений
        plt.plot(x_euler, y_euler, 'r--', label='Метод Эйлера', linewidth=2)
        plt.plot(x_improved, y_improved, 'b-', label='Усовершенствованный метод Эйлера', linewidth=2)
        plt.plot(x_milne, y_milne, 'g:', label='Метод Милна', linewidth=2, markersize=6)
        

        x_exact = np.linspace(x0, xn, 1000)
        y_exact = exact_func(x_exact, x0, y0)
        plt.plot(x_exact, y_exact, 'k-', label='Точное решение', linewidth=1, alpha=0.7)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Решение ОДУ: {eq["name"]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Анализ достижения требуемой точности
        print("\n" + "="*80)
        print("АНАЛИЗ ДОСТИЖЕНИЯ ТРЕБУЕМОЙ ТОЧНОСТИ")
        print("="*80)
        print(f"Требуемая точность ε = {epsilon:.2e}")
        
        # Проверяем точность
        accuracy_achieved = True

        errors = [
            ("Метод Эйлера", euler_runge_error),
            ("Усовершенствованный метод Эйлера", improved_euler_runge_error),
            ("Метод Милна", milne_exact_error)
        ]
        
        print(f"\nСравнение погрешностей с требуемой точностью:")
        for method_name, error in errors:
            if error <= epsilon:
                print(f"✓ {method_name}: точность ДОСТИГНУТА ({error:.2e} ≤ {epsilon:.2e})")
            else:
                print(f"✗ {method_name}: точность НЕ ДОСТИГНУТА ({error:.2e} > {epsilon:.2e})")
                accuracy_achieved = False
        
        print(f"\n{'='*80}")
        if accuracy_achieved:
            print("ВЫВОД: Требуемая точность достигнута всеми методами!")
        else:
            print("ВЫВОД: Требуемая точность НЕ достигнута некоторыми методами.")
            print("Рекомендация: уменьшите шаг интегрирования h и повторите вычисления.")
        print("="*80)


def validate_input(prompt: str, input_type: type, condition=None):
    """Валидация пользовательского ввода"""
    while True:
        try:
            value = input_type(input(prompt))
            if condition and not condition(value):
                print("Некорректное значение! Попробуйте снова.")
                continue
            return value
        except ValueError:
            print("Некорректный формат! Попробуйте снова.")


def main():
    solver = ODESolver()
    while True:
        print("\nДоступные уравнения:")
        for num, eq in solver.equations.items():
            print(f"{num}. {eq['name']}")
            print(f"   {eq['description']}")
        
        equation_num = validate_input(
            "\nВыберите номер уравнения (1-3): ",
            int,
            lambda x: x in solver.equations
        )
        
        print("\nВвод параметров:")
        x0 = validate_input("Начальная точка x₀: ", float)
        y0 = validate_input("Начальное условие y₀ = y(x₀): ", float)
        xn = validate_input("Конечная точка xₙ: ", float, lambda x: x > x0)
        h = validate_input("Шаг интегрирования h: ", float, lambda x: 0 < x < (xn - x0))
        epsilon = validate_input("Требуемая точность ε: ", float, lambda x: x > 0)

        try:
            solver.solve_and_analyze(equation_num, x0, y0, xn, h, epsilon)
        except Exception as e:
            print(f"Ошибка при вычислениях: {e}")
            print("Возможно, выбранные параметры привели к неустойчивости численного метода.")
        
        continue_work = input("\nХотите решить другое уравнение? (y/n): ").lower()
        if continue_work != 'y':
            break
    
    print("\nПрограмма завершена. Спасибо за использование!")


if __name__ == "__main__":
    main()
