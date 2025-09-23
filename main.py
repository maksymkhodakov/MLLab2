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

# Стратифікований CV, щоб у кожному фолді були всі класи
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# Абсолютні train_sizes (ВАЖЛИВО: <= макс. навчального розміру у фолді = 150 * 4/5 = 120)
LEARN_SIZES = np.array([60, 80, 100, 120], dtype=int)

# Сітка C для validation curve
C_RANGE = np.logspace(-3, 3, 7)

# ============================================================
# ЕТАП 1: ЗАВАНТАЖЕННЯ ДАНИХ
# ============================================================
print("ЕТАП 1: ЗАВАНТАЖЕННЯ ДАНИХ")
data = pd.read_csv("Iris.csv")
print("Розмірність датасету:", data.shape)
print(data.head(3).to_string(index=False))

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
# ЕТАП 3: РОЗБИТТЯ 60/20/20 (стратифіковано)
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
# ДОПОМІЖНІ: звіти, графіки, перевірки
# ============================================================
def overfit_report(name, y_tr_true, y_tr_pred, y_va_true, y_va_pred, gap_thr=0.05):
    f1_tr = f1_score(y_tr_true, y_tr_pred, average="macro")
    f1_va = f1_score(y_va_true, y_va_pred, average="macro")
    gap = f1_tr - f1_va
    print(f"[Overfit check] {name}: F1(train)={f1_tr:.3f}  F1(val)={f1_va:.3f}  GAP={gap:+.3f}")
    if gap > gap_thr:
        print("  ⚠ Перенавчання: train помітно кращий за val.")
    elif gap < -gap_thr:
        print("  ⚠ Аномалія/нестабільність: val кращий за train.")
    else:
        print("  ✅ Баланс train/val виглядає здорово.")


def plot_learning_curve_for(pipe, X_all, y_all, title):
    print(f"Побудова Learning Curve: {title}")
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X_all, y_all, cv=CV, scoring="f1_macro",
        train_sizes=LEARN_SIZES, n_jobs=None, error_score=np.nan
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


def plot_validation_curve_for(model, X_all, y_all, title, penalty):
    print(f"Побудова Validation Curve (C): {title}")
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    train_scores, val_scores = validation_curve(
        pipe, X_all, y_all, param_name="clf__C", param_range=C_RANGE,
        cv=CV, scoring="f1_macro", n_jobs=None, error_score=np.nan
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
    print(f"\n==================== {name} ====================")
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])

    print("Навчання моделі...")
    pipe.fit(X_train, y_train)

    y_pred_train = pipe.predict(X_train)
    y_pred_val = pipe.predict(X_val)
    y_pred_test = pipe.predict(X_test)

    print("\n--- ЗВІТ: TRAIN ---")
    print(classification_report(y_train, y_pred_train, target_names=le.classes_))
    print("--- ЗВІТ: VALIDATION ---")
    print(classification_report(y_val, y_pred_val, target_names=le.classes_))
    print("--- ЗВІТ: TEST ---")
    print(classification_report(y_test, y_pred_test, target_names=le.classes_))

    overfit_report(name, y_train, y_pred_train, y_val, y_pred_val)

    cm = confusion_matrix(y_test, y_pred_test)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_).plot(cmap="Blues")
    plt.title(f"Confusion Matrix (Test) — {name}")
    plt.tight_layout()
    plt.show()

    plot_learning_curve_for(pipe, X, y_encoded, f"Learning Curve — {name}")
    penalty = model.get_params()["penalty"]
    plot_validation_curve_for(model, X, y_encoded, f"Validation Curve — {name}", penalty=penalty)


# ============================================================
# ЕТАП 4: МОДЕЛІ (L2 і L1)
# ============================================================
print("\nЕТАП 4: МОДЕЛІ (L2, L1)")

# L2 — стабільний lbfgs
logreg_l2 = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)

# L1 — беремо liblinear (для маленьких датасетів сходиться швидко і стабільно)
logreg_l1 = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000, random_state=RANDOM_SEED)

# ============================================================
# ЕТАП 5: ОЦІНКА КОЖНОЇ МОДЕЛІ
# ============================================================
evaluate_model(logreg_l2, "Logistic Regression — L2")
evaluate_model(logreg_l1, "Logistic Regression — L1")
