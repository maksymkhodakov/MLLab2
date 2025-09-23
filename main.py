# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve, validation_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.pipeline import Pipeline

# ============================================================
# НАЛАШТУВАННЯ
# ============================================================
RANDOM_SEED = 42
SPLIT_TRAIN = 0.6
SPLIT_VAL = 0.2
SPLIT_TEST = 0.2

# Перевірка коректності часток
assert abs(SPLIT_TRAIN + SPLIT_VAL + SPLIT_TEST - 1.0) < 1e-9

# CV-спліттер: стратифікований, щоб у кожному фолді були всі класи
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# Для learning_curve: не беремо надто малі train_sizes, щоб у кожному фолді був кожен клас
LEARN_SIZES = np.linspace(0.3, 1.0, 8)  # 30% ... 100%

# Сітка C для validation_curve (регуляризація)
C_RANGE = np.logspace(-3, 3, 7)  # 0.001 ... 1000

# ============================================================
# ЕТАП 1: ЗАВАНТАЖЕННЯ ДАНИХ
# ============================================================
print("ЕТАП 1: ЗАВАНТАЖЕННЯ ДАНИХ")
data = pd.read_csv("Iris.csv")
print("Розмірність датасету:", data.shape)
print(data.head(3).to_string(index=False))

# Видаляємо службову колонку Id (якщо є)
if "Id" in data.columns:
    data = data.drop("Id", axis=1)
print("Після видалення Id -> колонки:", list(data.columns))

# ============================================================
# ЕТАП 2: ОЗНАКИ ТА ЦІЛЬ
# ============================================================
print("\nЕТАП 2: ФОРМУВАННЯ X та y")
X = data.drop("Species", axis=1)
y = data["Species"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Класи:", list(le.classes_))

# ============================================================
# ЕТАП 3: РОЗБИТТЯ ДАНИХ 60/20/20 (стратифіковано)
# ============================================================
print("\nЕТАП 3: РОЗБИТТЯ ДАНИХ (train/val/test = 60/20/20)")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=(1 - SPLIT_TRAIN),
    random_state=RANDOM_SEED, stratify=y_encoded
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=SPLIT_TEST / (SPLIT_VAL + SPLIT_TEST),
    random_state=RANDOM_SEED, stratify=y_temp
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


# ============================================================
# ДОПОМІЖНІ: навчання, метрики, графіки, overfit-звіт
# ============================================================
def overfit_report(name, y_tr_true, y_tr_pred, y_va_true, y_va_pred):
    """Звіт про overfitting/underfitting за F1-macro."""
    f1_tr = f1_score(y_tr_true, y_tr_pred, average="macro")
    f1_va = f1_score(y_va_true, y_va_pred, average="macro")
    gap = f1_tr - f1_va
    print(f"[Overfit check] {name}: F1(train)={f1_tr:.3f}  F1(val)={f1_va:.3f}  GAP={gap:+.3f}")
    if gap > 0.05:
        print("  ⚠ Можливе перенавчання (train значно кращий за val).")
    elif gap < -0.05:
        print("  ⚠ Можливе недонавчання/нестабільність (val кращий за train).")
    else:
        print("  ✅ Явного перенавчання/недонавчання не спостерігається.")


def plot_learning_curve_for(pipe, X_all, y_all, title):
    """Крива навчання із стратифікованим CV і без надто малих train_sizes."""
    print(f"Побудова Learning Curve: {title}")
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X_all, y_all, cv=CV, scoring="f1_macro",
        train_sizes=LEARN_SIZES
    )
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train F1")
    plt.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation F1")
    plt.xlabel("Кількість тренувальних зразків")
    plt.ylabel("F1 (macro)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_validation_curve_for(base_model, X_all, y_all, title, penalty):
    """
    Валідаційна крива по C для заданого penalty.
    Використовуємо пайплайн (scale + logreg).
    """
    print(f"Побудова Validation Curve (C) для: {title}")
    model = LogisticRegression(
        penalty=penalty,
        solver=("saga" if penalty == "l1" else "lbfgs"),
        max_iter=(5000 if penalty == "l1" else 1000),
        random_state=RANDOM_SEED
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    train_scores, val_scores = validation_curve(
        pipe, X_all, y_all, param_name="clf__C", param_range=C_RANGE,
        cv=CV, scoring="f1_macro"
    )
    plt.figure(figsize=(8, 5))
    plt.semilogx(C_RANGE, train_scores.mean(axis=1), "o-", label="Train F1")
    plt.semilogx(C_RANGE, val_scores.mean(axis=1), "o-", label="Validation F1")
    plt.xlabel("C (менше => сильніша регуляризація)")
    plt.ylabel("F1 (macro)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def evaluate_model(model, name):
    """
    Повний цикл: навчання (pipeline), звіти на train/val/test,
    confusion matrix, learning/validation curves, overfit-звіт.
    """
    print(f"\n==================== {name} ====================")
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])

    # Навчання
    print("Навчання моделі...")
    pipe.fit(X_train, y_train)

    # Прогнози для звітів
    y_pred_train = pipe.predict(X_train)
    y_pred_val = pipe.predict(X_val)
    y_pred_test = pipe.predict(X_test)

    print("\n--- ЗВІТ: TRAIN ---")
    print(classification_report(y_train, y_pred_train, target_names=le.classes_))
    print("--- ЗВІТ: VALIDATION ---")
    print(classification_report(y_val, y_pred_val, target_names=le.classes_))
    print("--- ЗВІТ: TEST ---")
    print(classification_report(y_test, y_pred_test, target_names=le.classes_))

    # Overfitting / underfitting check
    overfit_report(name, y_train, y_pred_train, y_val, y_pred_val)

    # Confusion Matrix (Test)
    cm = confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix (Test) — {name}")
    plt.tight_layout()
    plt.show()

    # Learning Curve
    plot_learning_curve_for(pipe, X, y_encoded, f"Learning Curve — {name}")

    # Validation Curve (по C)
    penalty = model.get_params()["penalty"]
    plot_validation_curve_for(model, X, y_encoded, f"Validation Curve — {name}", penalty=penalty)


# ============================================================
# ЕТАП 4: МОДЕЛІ (L2 і L1)
# ============================================================
print("\nЕТАП 4: МОДЕЛІ (L2, L1)")

# L2 (ridge-like): стабільний solver lbfgs
logreg_l2 = LogisticRegression(
    penalty="l2", solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED
)

# L1 (lasso-like): потрібен saga; збільшимо max_iter для надійної збіжності
logreg_l1 = LogisticRegression(
    penalty="l1", solver="saga", max_iter=5000, random_state=RANDOM_SEED
)

# ============================================================
# ЕТАП 5: ЗАПУСК ОЦІНКИ ДЛЯ КОЖНОЇ МОДЕЛІ
# ============================================================
evaluate_model(logreg_l2, "Logistic Regression — L2")
evaluate_model(logreg_l1, "Logistic Regression — L1")
