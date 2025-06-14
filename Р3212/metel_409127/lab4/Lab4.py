import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import sys
import os
import re

class ApproximationMethod:    
    def __init__(self):
        self.x = None
        self.y = None
        self.n = 0
        self.approximations = {}
        self.best_approximation = None
        
    def input_data(self, source='console'):
        if source == 'console':
            while True:
                try:
                    n = input("Введите количество точек (8-12): ")
                    self.n = int(n)
                    if not 8 <= self.n <= 12:
                        print("Ошибка: Количество точек должно быть от 8 до 12. Попробуйте еще раз.")
                        continue
                    
                    self.x = np.zeros(self.n)
                    self.y = np.zeros(self.n)
                    
                    print("Введите пары значений x и y:")
                    i = 0
                    while i < self.n:
                        try:
                            values = input(f"Точка {i+1} (x y): ").split()
                            if len(values) != 2:
                                print(f"Ошибка: Неверный формат ввода для точки {i+1}. Нужно ввести два числа, разделенных пробелом.")
                                continue
                            
                            x_val = float(values[0])
                            y_val = float(values[1])
                            self.x[i] = x_val
                            self.y[i] = y_val
                            i += 1 
                        except ValueError:
                            print(f"Ошибка: Введите числовые значения для точки {i+1}. Попробуйте еще раз.")
                    
                    return
                except ValueError:
                    print("Ошибка: Введите корректное числовое значение для количества точек. Попробуйте еще раз.")
        else:
            attempts = 0
            max_attempts = 3
            while attempts < max_attempts:
                try:
                    if not os.path.exists(source):
                        print(f"Ошибка: Файл {source} не найден. Попробуйте еще раз.")
                        source = input("Введите корректное имя файла: ")
                        attempts += 1
                        continue
                        
                    with open(source, 'r') as f:
                        lines = f.readlines()
                        
                    points = []
                    for line in lines:
                        matches = re.findall(r'-?\d+\.?\d*', line)
                        if len(matches) >= 2:
                            points.append((float(matches[0]), float(matches[1])))
                    
                    self.n = len(points)
                    if not 8 <= self.n <= 12:
                        print(f"Ошибка: В файле {source} содержится {self.n} точек, а должно быть от 8 до 12.")
                        source = input("Введите другое имя файла: ")
                        attempts += 1
                        continue
                        
                    self.x = np.array([point[0] for point in points])
                    self.y = np.array([point[1] for point in points])
                    return 
                
                except Exception as e:
                    print(f"Ошибка при чтении файла: {e}")
                    if attempts < max_attempts - 1:
                        source = input("Введите другое имя файла: ")
                    attempts += 1
            
            print(f"После {max_attempts} неудачных попыток чтения файла, переключаемся на ввод с консоли.")
            self.input_data('console')
    
    def approximate(self):
        if self.x is None or self.y is None or self.n == 0:
            raise ValueError("Сначала введите данные")
        
        if np.isnan(self.x).any() or np.isnan(self.y).any() or np.isinf(self.x).any() or np.isinf(self.y).any():
            raise ValueError("Данные содержат недопустимые значения")
        
        self._linear_approximation()
        self._polynomial_approximation(2)
        self._polynomial_approximation(3)
        self._exponential_approximation()
        self._logarithmic_approximation()
        self._power_approximation()
        
        self._find_best_approximation()
    
    def _linear_approximation(self):
        try:
            a, b = np.polyfit(self.x, self.y, 1)
            f_x = a * self.x + b
            errors = f_x - self.y
            mse = np.mean(errors**2) #среднеарифмитическое 
            rmse = np.sqrt(mse) 
            r = np.corrcoef(self.x, self.y)[0, 1]
            r_squared = r**2
            
            self.approximations['linear'] = {
                'type': 'linear',
                'params': {'a': a, 'b': b},
                'function': lambda x: a * x + b,
                'formula': f'f(x) = {a:.6f}*x + {b:.6f}',
                'f_x': f_x,
                'errors': errors,
                'rmse': rmse,
                'r': r,
                'r_squared': r_squared
            }
        except Exception as e:
            print(f"Ошибка при линейной аппроксимации: {e}")
            self.approximations['linear'] = None
    
    def _polynomial_approximation(self, degree):
        try:
            coeffs = np.polyfit(self.x, self.y, degree)
            f_x = np.polyval(coeffs, self.x)
            errors = f_x - self.y
            mse = np.mean(errors**2)
            rmse = np.sqrt(mse)
            ss_total = np.sum((self.y - np.mean(self.y))**2)
            ss_residual = np.sum(errors**2)
            r_squared = 1 - (ss_residual / ss_total)
            
            formula = "f(x) = "
            for i, coef in enumerate(coeffs):
                power = degree - i
                if power > 1:
                    formula += f"{coef:.6f}*x^{power} + "
                elif power == 1:
                    formula += f"{coef:.6f}*x + "
                else:
                    formula += f"{coef:.6f}"
            
            self.approximations[f'polynomial_{degree}'] = {
                'type': f'polynomial_{degree}',
                'params': {'coeffs': coeffs},
                'function': lambda x: np.polyval(coeffs, x),
                'formula': formula,
                'f_x': f_x,
                'errors': errors,
                'rmse': rmse,
                'r_squared': r_squared
            }
        except Exception as e:
            print(f"Ошибка при полиномиальной аппроксимации степени {degree}: {e}")
            self.approximations[f'polynomial_{degree}'] = None
    
    def _exponential_approximation(self):
        try:
            if np.any(self.y <= 0):
                raise ValueError("Экспоненциальная аппроксимация невозможна для отрицательных значений y")
            
            ln_y = np.log(self.y)
            b, ln_a = np.polyfit(self.x, ln_y, 1)
            a = np.exp(ln_a)
            
            def exp_func(x, a, b):
                return a * np.exp(b * x)
            
            try:
                params, _ = optimize.curve_fit(exp_func, self.x, self.y, p0=[a, b])
                a, b = params
            except RuntimeError:
                pass
            
            f_x = a * np.exp(b * self.x)
            errors = f_x - self.y
            mse = np.mean(errors**2)
            rmse = np.sqrt(mse)
            ss_total = np.sum((self.y - np.mean(self.y))**2)
            ss_residual = np.sum(errors**2)
            r_squared = 1 - (ss_residual / ss_total)
            
            self.approximations['exponential'] = {
                'type': 'exponential',
                'params': {'a': a, 'b': b},
                'function': lambda x: a * np.exp(b * x),
                'formula': f'f(x) = {a:.6f}*exp({b:.6f}*x)',
                'f_x': f_x,
                'errors': errors,
                'rmse': rmse,
                'r_squared': r_squared
            }
        except Exception as e:
            print(f"Ошибка при экспоненциальной аппроксимации: {e}")
            self.approximations['exponential'] = None
    
    def _logarithmic_approximation(self):
        try:
            if np.any(self.x <= 0):
                raise ValueError("Логарифмическая аппроксимация невозможна для отрицательных значений x")
            
            ln_x = np.log(self.x)
            a, b = np.polyfit(ln_x, self.y, 1)
            
            f_x = a * np.log(self.x) + b
            errors = f_x - self.y
            mse = np.mean(errors**2)
            rmse = np.sqrt(mse)
            ss_total = np.sum((self.y - np.mean(self.y))**2)
            ss_residual = np.sum(errors**2)
            r_squared = 1 - (ss_residual / ss_total)
            
            self.approximations['logarithmic'] = {
                'type': 'logarithmic',
                'params': {'a': a, 'b': b},
                'function': lambda x: a * np.log(x) + b,
                'formula': f'f(x) = {a:.6f}*ln(x) + {b:.6f}',
                'f_x': f_x,
                'errors': errors,
                'rmse': rmse,
                'r_squared': r_squared
            }
        except Exception as e:
            print(f"Ошибка при логарифмической аппроксимации: {e}")
            self.approximations['logarithmic'] = None
    
    def _power_approximation(self):
        try:
            if np.any(self.x <= 0) or np.any(self.y <= 0):
                raise ValueError("Степенная аппроксимация невозможна для отрицательных значений x или y")
            
            ln_x = np.log(self.x)
            ln_y = np.log(self.y)
            b, ln_a = np.polyfit(ln_x, ln_y, 1)
            a = np.exp(ln_a)
            
            f_x = a * np.power(self.x, b)
            errors = f_x - self.y
            mse = np.mean(errors**2)
            rmse = np.sqrt(mse)
            ss_total = np.sum((self.y - np.mean(self.y))**2)
            ss_residual = np.sum(errors**2)
            r_squared = 1 - (ss_residual / ss_total)
            
            self.approximations['power'] = {
                'type': 'power',
                'params': {'a': a, 'b': b},
                'function': lambda x: a * np.power(x, b),
                'formula': f'f(x) = {a:.6f}*x^{b:.6f}',
                'f_x': f_x,
                'errors': errors,
                'rmse': rmse,
                'r_squared': r_squared
            }
        except Exception as e:
            print(f"Ошибка при степенной аппроксимации: {e}")
            self.approximations['power'] = None
            
    def _find_best_approximation(self):
        min_rmse = float('inf')
        for appr_type, data in self.approximations.items():
            if data is not None and data['rmse'] < min_rmse:
                min_rmse = data['rmse']
                self.best_approximation = appr_type
    
    def print_results(self, output='console'):
        if not self.approximations:
            raise ValueError("Сначала выполните аппроксимацию")
        
        results = ["\n" + "="*80, "РЕЗУЛЬТАТЫ АППРОКСИМАЦИИ", "="*80]
        
        results.append(f"\nИсходные данные (всего {self.n} точек):")
        for i in range(self.n):
            results.append(f"Точка {i+1}: x = {self.x[i]:.6f}, y = {self.y[i]:.6f}")
        
        results.append("\nРезультаты аппроксимации:")
        for appr_type, data in self.approximations.items():
            if data is None:
                results.append(f"\n{appr_type.upper()}: не удалось выполнить аппроксимацию")
                continue
                
            results.append(f"\n{appr_type.upper()}:")
            results.append(f"Формула: {data['formula']}")
            results.append(f"Среднеквадратическое отклонение (RMSE): {data['rmse']:.6f}")
            
            if appr_type == 'linear':
                r = data['r']
                results.append(f"Коэффициент корреляции Пирсона: {r:.6f}")
                r_abs = abs(r)
                if r_abs < 0.3:
                    corr = "очень слабая"
                elif r_abs < 0.5:
                    corr = "слабая"
                elif r_abs < 0.7:
                    corr = "средняя"
                elif r_abs < 0.9:
                    corr = "сильная"
                else:
                    corr = "очень сильная"
                direction = "положительная" if r > 0 else "отрицательная"
                results.append(f"Интерпретация: {corr} {direction} корреляция")
            
            r_squared = data['r_squared']
            results.append(f"Коэффициент детерминации (R^2): {r_squared:.6f}")
            if r_squared < 0.3:
                determ = "Модель объясняет малую часть дисперсии данных"
            elif r_squared < 0.5:
                determ = "Модель объясняет некоторую часть дисперсии данных"
            elif r_squared < 0.7:
                determ = "Модель хорошо объясняет дисперсию данных"
            else:
                determ = "Модель очень хорошо объясняет дисперсию данных"
            results.append(f"Интерпретация: {determ}")
            
            results.append("\nТочки аппроксимации:")
            for i in range(min(5, self.n)):  # Выводим только первые 5 точек для краткости
                error = data['errors'][i]
                results.append(f"Точка {i+1}: x = {self.x[i]:.6f}, y = {self.y[i]:.6f}, f(x) = {data['f_x'][i]:.6f}, e = {error:.6f}")
            if self.n > 5:
                results.append("...")
        
        results.append("\n" + "="*80)
        results.append(f"НАИЛУЧШАЯ АППРОКСИМАЦИЯ: {self.best_approximation.upper()}")
        results.append(f"Формула: {self.approximations[self.best_approximation]['formula']}")
        results.append(f"Среднеквадратическое отклонение (RMSE): {self.approximations[self.best_approximation]['rmse']:.6f}")
        results.append("="*80)
        
        if output == 'console':
            for line in results:
                print(line)
        else:
            with open(output, 'w') as f:
                for line in results:
                    f.write(line + "\n")
            print(f"Результаты сохранены в файл {output}")
    
    def plot_graphs(self, save_to_file=None):
        if not self.approximations:
            raise ValueError("Сначала выполните аппроксимацию")
        
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()
        
        x_min, x_max = np.min(self.x), np.max(self.x)
        y_min, y_max = np.min(self.y), np.max(self.y)
        
        x_range, y_range = x_max - x_min, y_max - y_min
        x_min, x_max = x_min - 0.2 * x_range, x_max + 0.2 * x_range
        y_min, y_max = y_min - 0.2 * y_range, y_max + 0.2 * y_range
        
        x_grid = np.linspace(x_min, x_max, 1000)
        
        type_names = {
            'linear': 'Линейная',
            'polynomial_2': 'Полиномиальная 2-й степени',
            'polynomial_3': 'Полиномиальная 3-й степени',
            'exponential': 'Экспоненциальная',
            'logarithmic': 'Логарифмическая',
            'power': 'Степенная'
        }
        
        for i, (appr_type, data) in enumerate(self.approximations.items()):
            if data is None:
                axs[i].set_title(f"{type_names.get(appr_type, appr_type)}: не удалось")
                continue
                
            try:
                if appr_type in ['logarithmic', 'power']:
                    valid_x = x_grid[x_grid > 0]
                    y_grid = data['function'](valid_x)
                    axs[i].plot(valid_x, y_grid, 'r-', linewidth=2)
                else:
                    y_grid = data['function'](x_grid)
                    axs[i].plot(x_grid, y_grid, 'r-', linewidth=2)
                
                axs[i].scatter(self.x, self.y, color='blue', s=30)
                axs[i].set_title(f"{type_names.get(appr_type, appr_type)}: RMSE={data['rmse']:.4f}")
                axs[i].set_xlabel('x')
                axs[i].set_ylabel('y')
                axs[i].text(0.05, 0.95, data['formula'], transform=axs[i].transAxes, 
                          fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                axs[i].set_xlim(x_min, x_max)
                axs[i].set_ylim(y_min, y_max)
                
                if appr_type == self.best_approximation:
                    axs[i].set_facecolor('#e0ffe0')
                    axs[i].set_title(f"{type_names.get(appr_type, appr_type)}: НАИЛУЧШАЯ! RMSE={data['rmse']:.4f}")
            except Exception as e:
                print(f"Ошибка при построении графика для {appr_type}: {e}")
        
        fig.tight_layout()
        
        if save_to_file:
            plt.savefig(save_to_file, dpi=300)
            print(f"Графики сохранены в файл {save_to_file}")
        else:
            plt.show()

def main():    
    approx = ApproximationMethod()
    while True:
        source = input("Выберите источник данных (1 - консоль, 2 - файл): ")
        if source == '1':
            approx.input_data('console')
            break
        elif source == '2':
            filename = input("Введите имя файла с данными: ")
            approx.input_data(filename)
            break
        else:
            print("Неверный выбор. Выберите 1 или 2.")
    
    try:
        approx.approximate()
    except Exception as e:
        print(f"Ошибка при аппроксимации: {e}")
        return
    
    while True:
        output = input("Выберите способ вывода результатов (1 - консоль, 2 - файл, 3 - оба): ")
        if output == '1':
            approx.print_results('console')
            break
        elif output == '2':
            while True:
                try:
                    filename = input("Введите имя файла для сохранения результатов: ")
                    approx.print_results(filename)
                    break
                except Exception as e:
                    print(f"Ошибка при сохранении результатов: {e}. Попробуйте другое имя файла.")
            break
        elif output == '3':
            while True:
                try:
                    filename = input("Введите имя файла для сохранения результатов: ")
                    approx.print_results('console')
                    approx.print_results(filename)
                    break
                except Exception as e:
                    print(f"Ошибка при сохранении результатов: {e}. Попробуйте другое имя файла.")
            break
        else:
            print("Неверный выбор. Выберите 1, 2 или 3.")
    
    while True:
        graph_option = input("Построить графики? (y/n): ").lower()
        if graph_option == 'y':
            while True:
                save_option = input("Сохранить графики в файл? (y/n): ").lower()
                if save_option == 'y':
                    while True:
                        try:
                            image_file = input("Введите имя файла для сохранения графиков: ")
                            approx.plot_graphs(save_to_file=image_file)
                            break
                        except Exception as e:
                            print(f"Ошибка при сохранении графиков: {e}. Попробуйте другое имя файла.")
                    break
                elif save_option == 'n':
                    try:
                        approx.plot_graphs()
                    except Exception as e:
                        print(f"Ошибка при построении графиков: {e}")
                    break
                else:
                    print("Неверный выбор. Введите 'y' или 'n'.")
            break
        elif graph_option == 'n':
            break
        else:
            print("Неверный выбор. Введите 'y' или 'n'.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        sys.exit(1)
