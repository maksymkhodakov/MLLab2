import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    learning_curve,
    validation_curve,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)
from sklearn.pipeline import Pipeline

# ============================================================
# НАЛАШТУВАННЯ (винесені у змінні)
# ============================================================
RANDOM_SEED = 42  # фіксуємо "випадковість" для відтворюваності результатів
SPLIT_TRAIN = 0.60  # частка тренувальної вибірки
SPLIT_VAL = 0.20  # частка валідаційної
SPLIT_TEST = 0.20  # частка тестової
assert abs(SPLIT_TRAIN + SPLIT_VAL + SPLIT_TEST - 1.0) < 1e-9, "Частки сплітів мусять сумуватися до 1.0"

# Для validation_curve: сітка значень C (лог-шкала від 0.001 до 1000)
C_RANGE = np.logspace(-3, 3, 7)

# ============================================================
# ЕТАП 1: ЗАВАНТАЖЕННЯ ДАНИХ
# ============================================================
print("ЕТАП 1: ЗАВАНТАЖЕННЯ ДАНИХ")
data = pd.read_csv("Iris.csv")  # якщо файл у тій самій теці; інакше вкажи повний шлях
print("Розмірність датасету:", data.shape)
print("Перші рядки:\n", data.head(3).to_string(index=False))

# Колонка Id — просто індекс; у моделі не використовується → видаляємо
if "Id" in data.columns:
    data = data.drop("Id", axis=1)
print("Після видалення Id → колонки:", list(data.columns))

# (за бажанням) Швидкий EDA: перевірити класи
print("\nРозподіл класів:\n", data["Species"].value_counts())

# ============================================================
# ЕТАП 2: ОЗНАКИ ТА ЦІЛЬ
# ============================================================
print("\nЕТАП 2: ФОРМУВАННЯ X та y")
X = data.drop("Species", axis=1)  # 4 числові ознаки: SepalLength, SepalWidth, PetalLength, PetalWidth
y = data["Species"]  # мітки класів як рядки

n_samples = len(X)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
max_train_size = int(n_samples * (CV.n_splits - 1) / CV.n_splits)

# Для learning_curve: беремо "абсолютні" розміри, достатньо великі для трьох класів
LEARN_SIZES = np.linspace(0.5, 1.0, 4) * max_train_size
LEARN_SIZES = LEARN_SIZES.astype(int)

print("Автоматичні розміри підвибірки train:", LEARN_SIZES)

# Перетворюємо рядкові мітки у числа (0/1/2) — так зручніше для моделей
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Класи (порядок кодування):", list(le.classes_))

# кореляційна heatmap ознак
plt.figure(figsize=(6, 5))
sns.heatmap(pd.DataFrame(X, columns=X.columns).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Кореляційна матриця ознак (Iris)")
plt.tight_layout()
plt.show()

# (опційно) Pairplot — гарно видно роздільність класів у просторах 2D
sns.pairplot(pd.concat([X.reset_index(drop=True), pd.Series(le.inverse_transform(y_encoded), name="Species")], axis=1),
             hue="Species", diag_kind="hist")
plt.suptitle("Pairplot ознак Iris (кольором — класи)", y=1.02)
plt.show()

# ============================================================
# ЕТАП 3: РОЗБИТТЯ ДАНИХ 60/20/20 (СТРАТИФІКОВАНО)
# ============================================================
print("\nЕТАП 3: РОЗБИТТЯ ДАНИХ (train/val/test = 60/20/20)")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded,
    test_size=(1 - SPLIT_TRAIN),  # 40% підуть у тимчасовий набір
    random_state=RANDOM_SEED,
    stratify=y_encoded  # важливо: зберігати пропорції класів
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=SPLIT_TEST / (SPLIT_VAL + SPLIT_TEST),  # 0.2 / (0.2+0.2) = 0.5
    random_state=RANDOM_SEED,
    stratify=y_temp
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


# ============================================================
# ДОПОМІЖНІ ФУНКЦІЇ: звіти, графіки, перевірка overfit
# ============================================================
def overfit_report(name, y_tr_true, y_tr_pred, y_va_true, y_va_pred, gap_thr=0.05):
    """
    Проста перевірка перенавчання:
    - Рахуємо macro-F1 на train та val.
    - Дивимося на різницю (gap). Якщо F1(train) суттєво > F1(val), є ризик overfitting.
    Поріг gap_thr=0.05 — евристичний (можна підкрутити).
    """
    f1_tr = f1_score(y_tr_true, y_tr_pred, average="macro")
    f1_va = f1_score(y_va_true, y_va_pred, average="macro")
    gap = f1_tr - f1_va
    print(f"[Overfit check] {name}: F1(train)={f1_tr:.3f}  F1(val)={f1_va:.3f}  GAP={gap:+.3f}")
    if gap > gap_thr:
        print("  ⚠ Перенавчання: train значно кращий за val (можливо, замалий C/слабка регуляризація).")
    elif gap < -gap_thr:
        print("  ⚠ Дивна ситуація: val кращий за train (нестабільність/випадковість).")
    else:
        print("  ✅ Баланс train/val виглядає здорово.")


def plot_learning_curve_for(pipe, X_all, y_all, title):
    """
    Крива навчання: показує F1_macro на train/val при зростанні розміру train-set.
    - Якщо train F1 >> val F1 при великих наборах → перенавчання.
    - Якщо обидві криві низькі → недонавчання (модель занадто проста).
    """
    print(f"Побудова Learning Curve: {title}")
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X_all, y_all,
        cv=CV,
        scoring="f1_macro",
        train_sizes=LEARN_SIZES,  # абсолютні розміри (60,80,100,120) -> стабільність класів
        n_jobs=None,
        error_score=np.nan  # якщо якийсь фолд зламається, не падаємо (але ми це попередили)
    )
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, np.nanmean(train_scores, axis=1), "o-", label="Train F1")
    plt.plot(train_sizes, np.nanmean(val_scores, axis=1), "o-", label="Validation F1")
    plt.xlabel("Кількість тренувальних зразків (абс.)")
    plt.ylabel("F1 (macro)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_validation_curve_for(model, X_all, y_all, title):
    """
    Валідаційна крива по C: як змінюється F1_macro при різній силі регуляризації.
    - Ліва частина (малий C): сильна регуляризація → можливо underfit.
    - Права частина (великий C): слабка регуляризація → ризик overfit.
    - Шукаємо пік Validation F1 (оптимальний компроміс).
    """
    print(f"Побудова Validation Curve (C): {title}")
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    train_scores, val_scores = validation_curve(
        pipe, X_all, y_all,
        param_name="clf__C",
        param_range=C_RANGE,
        cv=CV,
        scoring="f1_macro",
        n_jobs=None,
        error_score=np.nan
    )
    plt.figure(figsize=(8, 5))
    plt.semilogx(C_RANGE, np.nanmean(train_scores, axis=1), "o-", label="Train F1")
    plt.semilogx(C_RANGE, np.nanmean(val_scores, axis=1), "o-", label="Validation F1")
    plt.xlabel("C (менше => сильніша регуляризація)")
    plt.ylabel("F1 (macro)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def evaluate_model(model, name):
    """
    Повний цикл оцінки однієї моделі логістичної регресії:
    - Створюємо пайплайн: StandardScaler (масштабування ознак) + LogisticRegression.
      Масштабування критично важливе для моделей з регуляризацією, бо всі фічі
      опиняються в порівнюваних масштабах, а штраф йде по вагам.
    - Навчаємо тільки на train.
    - Рахуємо звіти на train/val/test.
    - Будуємо матрицю плутанини (confusion matrix) на тесті.
    - Друкуємо звіт по overfit/underfit.
    - Малюємо Learning Curve та Validation Curve.
    """
    print(f"\n==================== {name} ====================")

    # Пайплайн з масштабуванням — щоби уникнути data leakage робимо fit тільки всередині pipeline
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])

    # Навчання
    print("Навчання моделі...")
    pipe.fit(X_train, y_train)

    # Прогнози
    y_pred_train = pipe.predict(X_train)
    y_pred_val = pipe.predict(X_val)
    y_pred_test = pipe.predict(X_test)

    # Повні звіти по метриках (precision/recall/F1/support) для кожного класу + macro avg
    print("\n--- ЗВІТ: TRAIN ---")
    print(classification_report(y_train, y_pred_train, target_names=le.classes_))
    print("--- ЗВІТ: VALIDATION ---")
    print(classification_report(y_val, y_pred_val, target_names=le.classes_))
    print("--- ЗВІТ: TEST ---")
    print(classification_report(y_test, y_pred_test, target_names=le.classes_))

    # Перевірка на перенавчання/недонавчання
    overfit_report(name, y_train, y_pred_train, y_val, y_pred_val)

    # Матриця плутанини на тесті: де саме модель помиляється (плутанина між versicolor та virginica — класика)
    cm = confusion_matrix(y_test, y_pred_test)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_).plot(cmap="Blues")
    plt.title(f"Confusion Matrix (Test) — {name}")
    plt.tight_layout()
    plt.show()

    # Крива навчання — чи зближаються Train/Val при збільшенні прикладів?
    plot_learning_curve_for(pipe, X, y_encoded, f"Learning Curve — {name}")

    # Крива валідації по C — де максимум якості на валідації?
    plot_validation_curve_for(model, X, y_encoded, f"Validation Curve — {name}")


# ============================================================
# ЕТАП 4: ОГОЛОШЕННЯ МОДЕЛЕЙ (L2 та L1) + ПОЯСНЕННЯ ПАРАМЕТРІВ
# ============================================================
print("\nЕТАП 4: МОДЕЛІ (L2, L1)")

# L2 (ридж-подібна регул.) — стабільний solver 'lbfgs' для багатокласовості
# max_iter=1000: ліміт ітерацій оптимізатора (запас, щоб збігся)
logreg_l2 = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    random_state=RANDOM_SEED
    # multi_class НЕ задаємо — залишаємо за замовчуванням (щоб уникнути FutureWarning)
)

# L1 (лассо-подібна регул.) — робить ваги розрідженими (деякі = 0) -> відбір ознак
# Для маленьких датасетів стабільний solver 'liblinear' (на відміну від 'saga', який іноді попереджає про збіжність)
logreg_l1 = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    max_iter=1000,
    random_state=RANDOM_SEED
)

# ============================================================
# ЕТАП 5: ЗАПУСК ОЦІНКИ КОЖНОЇ МОДЕЛІ
# ============================================================
evaluate_model(logreg_l2, "Logistic Regression — L2")
evaluate_model(logreg_l1, "Logistic Regression — L1")
