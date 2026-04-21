import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

TRAIN_PATH = "train"
TEST_PATH = "test"
CATEGORIES = ["winter", "summer", "spring", "fall"]


# Переводить зображення в HSV, додає просторову інформацію та аналіз текстур
def extract_features(image, bins=(8, 8, 8)):
    features = []

    # 1. Просторова інформація (розбиваємо кадр на сітку 2x2)
    h, w = image.shape[:2]
    h_half, w_half = h // 2, w // 2

    quadrants = [
        image[0:h_half, 0:w_half],  # Верх-ліво
        image[0:h_half, w_half:w],  # Верх-право
        image[h_half:h, 0:w_half],  # Низ-ліво
        image[h_half:h, w_half:w]  # Низ-право
    ]

    # Витягуємо колірну гістограму для кожного з 4 секторів окремо
    for quad in quadrants:
        hsv = cv2.cvtColor(quad, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        features.extend(hist.flatten())

    #Аналіз текстур (Гістограма магнітуди градієнтів Собеля)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Знаходимо різкі переходи (контури гілок, листя, снігу)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))

    # Будуємо гістограму для текстур (32 корзини)
    texture_hist = cv2.calcHist([magnitude], [0], None, [32], [0, 256])
    cv2.normalize(texture_hist, texture_hist)
    features.extend(texture_hist.flatten())

    return np.array(features)


# Завантажує дані з розсортованих папок для навчання.
def load_train_data(base_path):
    X = []
    y = []
    for label_idx, category in enumerate(CATEGORIES):
        folder_path = os.path.join(base_path, category)
        print(f"Завантаження тренувальних зображень з: {folder_path}")

        if not os.path.exists(folder_path):
            continue

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)

            if image is not None:
                image = cv2.resize(image, (256, 256))
                features = extract_features(image)
                X.append(features)
                y.append(label_idx)

    return np.array(X), np.array(y)


# Робить прогнози для папки test, де зображення лежать без сортування.
def predict_unlabeled_folder(model, folder_path):
    print(f"\nАналіз невідомих зображень з папки '{folder_path}'")
    if not os.path.exists(folder_path):
        print(f"Папку '{folder_path}' не знайдено.")
        return

    files = os.listdir(folder_path)
    if len(files) == 0:
        print("Папка порожня.")
        return

    for filename in files:
        img_path = os.path.join(folder_path, filename)
        image = cv2.imread(img_path)

        if image is not None:
            small_image = cv2.resize(image, (256, 256))
            features = extract_features(small_image)

            # Робимо прогноз
            prediction_idx = model.predict([features])[0]
            predicted_season = CATEGORIES[prediction_idx]

            print(f"Файл: {filename:20} --> Прогноз: {predicted_season.upper()}")
        else:
            print(f"Файл: {filename:20} --> Не вдалося прочитати зображення")


if __name__ == "__main__":
    print("--- Підготовка даних ---")
    X, y = load_train_data(TRAIN_PATH)

    if len(X) == 0:
        print("Помилка: Тренувальні дані не знайдено!")
        exit()

    # Відкушуємо 20% від train для створення звіту про точність
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Використання ансамблевого методу Random Forest
    print("\n--- Навчання моделі Random Forest ---")
    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("\n--- Оцінка точності (на 20% відкладених даних) ---")
    val_predictions = model.predict(X_val)
    print(classification_report(y_val, val_predictions, target_names=CATEGORIES))

    # Збереження моделі
    joblib.dump(model, "season_rf_model.pkl")
    print("Модель успішно збережено у 'season_rf_model.pkl'")

    # Аналіз папки test
    predict_unlabeled_folder(model, TEST_PATH)