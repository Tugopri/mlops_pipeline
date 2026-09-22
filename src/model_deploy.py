import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.ft_engineering import prepare_train_test

df = pd.read_csv("C:\M52.0\mlops_pipeline\Base_de_datos1.cs") 
print(df.shape) 
df.head()

X_train, X_test, y_train, y_test, preprocessor = prepare_train_test(
    df,
    target="Pago_atiempo"
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

def build_model(model):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=8
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}

def summarize_classification(y_true, y_pred, model_name):
    return {
        "Modelo": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "F1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }

results = []
trained_models = {}

for name, model in models.items():

    pipeline = build_model(model)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    result = summarize_classification(
        y_test,
        y_pred,
        name
    )

    results.append(result)

    trained_models[name] = pipeline

    print("=" * 60)
    print(name)
    print("=" * 60)
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

results_df = pd.DataFrame(results)

results_df

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1"
]

results_plot = results_df.set_index("Modelo")[metrics]

results_plot.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Comparación de modelos de clasificación")
plt.ylabel("Valor")
plt.xlabel("Modelo")
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.legend(title="Métrica")
plt.tight_layout()
plt.show()

for name, pipeline in trained_models.items():

    y_pred = pipeline.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{name}")
    print(cm)

from sklearn.metrics import ConfusionMatrixDisplay

for name, pipeline in trained_models.items():

    y_pred = pipeline.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred
    )

    plt.title(f"Matriz de confusión - {name}")
    plt.tight_layout()
    plt.show()

results_df.to_csv(
    "resultados_modelos.csv",
    index=False
)