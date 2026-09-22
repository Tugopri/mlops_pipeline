import pandas as pd

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

from ft_engineering import prepare_train_test


def build_model(model, preprocessor):
    """
    Construye un pipeline combinando el preprocesamiento
    con el modelo de clasificación.
    """

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def summarize_classification(y_true, y_pred, model_name):
    """
    Resume las métricas principales de clasificación.
    """

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


def train_and_evaluate(df, target="Pago_atiempo"):
    """
    Entrena y evalúa los modelos de clasificación.
    """

    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test(
        df,
        target=target
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    trained_models = {}

    for name, model in models.items():

        pipeline = build_model(
            model,
            preprocessor
        )

        pipeline.fit(
            X_train,
            y_train
        )

        y_pred = pipeline.predict(X_test)

        summary = summarize_classification(
            y_test,
            y_pred,
            name
        )

        results.append(summary)

        trained_models[name] = pipeline

        print("=" * 60)
        print(name)
        print("=" * 60)

        print(
            classification_report(
                y_test,
                y_pred,
                zero_division=0
            )
        )

        print("Matriz de confusión:")
        print(confusion_matrix(y_test, y_pred))

    results_df = pd.DataFrame(results)

    return (
        results_df,
        trained_models,
        X_train,
        X_test,
        y_train,
        y_test
    )