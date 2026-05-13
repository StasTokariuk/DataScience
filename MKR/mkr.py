import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
matplotlib.use('TkAgg')


# клас для реалізації рекурентного експоненційного згладжування
class RecursiveEMAFilter:

    def __init__(self, alpha: float):
        if not 0 < alpha <= 1:
            raise ValueError("Коефіцієнт згладжування alpha має бути в межах (0, 1]")
        self.alpha = alpha
        self.smoothed_data = np.array([])

    # застосовує рекурентне згладжування до масиву вимірів.
    def fit_transform(self, measurements: np.ndarray) -> np.ndarray:
        if len(measurements) == 0:
            return self.smoothed_data

        self.smoothed_data = np.zeros_like(measurements)
        self.smoothed_data[0] = measurements[0]

        # Рекурентне обчислення
        for t in range(1, len(measurements)):
            self.smoothed_data[t] = (self.alpha * measurements[t] +
                                     (1 - self.alpha) * self.smoothed_data[t - 1])

        return self.smoothed_data


def main():
    # Генерація вибірки
    np.random.seed(42)
    time_steps = np.linspace(0, 12, 300)

    # Ідеальний сигнал дві гармоніки
    true_signal = np.sin(time_steps) + 0.5 * np.cos(2 * time_steps)

    # Додаємо Гауссівський шум
    noise_std = 0.45
    noisy_measurements = true_signal + np.random.normal(0, noise_std, size=len(time_steps))

    # Ініціалізація фільтрів з різними коефіцієнтами
    filter_strong = RecursiveEMAFilter(alpha=0.1)
    filter_optimal = RecursiveEMAFilter(alpha=0.3)

    # Застосування фільтрів
    smoothed_strong = filter_strong.fit_transform(noisy_measurements)
    smoothed_optimal = filter_optimal.fit_transform(noisy_measurements)

    # Аналіз похибки
    # Порівнюємо відфільтровані дані з ідеальним сигналом
    mse_raw = mean_squared_error(true_signal, noisy_measurements)
    mse_strong = mean_squared_error(true_signal, smoothed_strong)
    mse_optimal = mean_squared_error(true_signal, smoothed_optimal)

    print(f"MSE сирих зашумлених даних: {mse_raw:.4f}")
    print(f"MSE після сильного згладжування (alpha=0.1): {mse_strong:.4f}")
    print(f"MSE після оптимального згладжування (alpha=0.3): {mse_optimal:.4f}")

    # 4. Візуалізація
    plt.figure(figsize=(12, 6))

    plt.scatter(time_steps, noisy_measurements, color='lightgray', s=15,
                label=f'Зашумлені виміри (MSE={mse_raw:.2f})')
    plt.plot(time_steps, true_signal, color='black', linestyle='--', linewidth=1.5,
             label='Еталонний сигнал')

    plt.plot(time_steps, smoothed_strong, color='red', linewidth=2,
             label=f'Згладжування alpha=0.1 (MSE={mse_strong:.2f})')
    plt.plot(time_steps, smoothed_optimal, color='blue', linewidth=2.5,
             label=f'Згладжування alpha=0.3 (MSE={mse_optimal:.2f})')

    plt.title('Рекурентне згладжування вибірки з оцінкою MSE')
    plt.xlabel('Час ($t$)')
    plt.ylabel('Амплітуда')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
