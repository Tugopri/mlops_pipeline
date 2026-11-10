
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    average_precision_score
)

from src.ft_engineering import prepare_train_test


def build_model(model, preprocessor):
    """
    Construye un pipeline combinando el preprocesamiento
    con el modelo de clasificación.
    """

    return Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", model)
    ])


def summarize_classification(y_true, y_pred, y_proba, model_name):
    """
    Resume las métricas principales de clasificación.
    """

    return {
        "Modelo": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0
        ),
        "F1": f1_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0
        ),
        "PR-AUC": average_precision_score(
            (y_true == 0).astype(int),
            y_proba
        )
    }


def train_and_evaluate(df, target="Pago_atiempo"):
    """
    Entrena, compara y evalúa modelos de clasificación.
    """

    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test(
        df,
        target=target
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
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

        # Obtener probabilidad de incumplimiento (clase 0)
        classes = list(pipeline.classes_)
        class_zero_index = classes.index(0)

        y_proba = pipeline.predict_proba(X_test)[
            :, class_zero_index
        ]

        summary = summarize_classification(
            y_test,
            y_pred,
            y_proba,
            name
        )

        results.append(summary)

        trained_models[name] = pipeline

        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        print("\nReporte de clasificación:")
        print(
            classification_report(
                y_test,
                y_pred,
                zero_division=0
            )
        )

        print("\nMatriz de confusión:")
        print(confusion_matrix(y_test, y_pred))


    # Análisis de importancia de variables del Random Forest
    print("\n" + "=" * 60)
    print("IMPORTANCIA DE VARIABLES - RANDOM FOREST")
    print("=" * 60)

    random_forest = trained_models["Random Forest"]

    preprocessor = random_forest.named_steps["preprocessor"]
    classifier = random_forest.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    importances = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "Variable": feature_names,
        "Importancia": importances
    })

    importance_df = importance_df.sort_values(
        by="Importancia",
        ascending=False
    ).reset_index(drop=True)

    print(importance_df.head(20).to_string(index=False))

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="PR-AUC",
        ascending=False
    ).reset_index(drop=True)

    return (
        results_df,
        trained_models,
        X_train,
        X_test,
        y_train,
        y_test
    )


def save_models(trained_models, output_dir="models"):
    """
    Guarda los modelos entrenados en formato joblib.
    """

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    for name, model in trained_models.items():

        filename = (
            name.lower()
            .replace(" ", "_")
            + ".joblib"
        )

        joblib.dump(
            model,
            output_path / filename
        )

        print(f"Modelo guardado: {output_path / filename}")
