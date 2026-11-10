
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from src.model_deploy import deploy_model


def compare_models(results_df):
    """
    Genera una comparación gráfica de las métricas
    obtenidas por los modelos.
    """

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "PR-AUC"
    ]

    available_metrics = [
        metric
        for metric in metrics
        if metric in results_df.columns
    ]

    plot_data = results_df.set_index("Modelo")[
        available_metrics
    ]

    ax = plot_data.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title("Comparación de modelos de clasificación")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Métrica")
    ax.set_ylim(0, 1)

    plt.xticks(rotation=0)
    plt.legend(title="Métrica")
    plt.tight_layout()
    plt.show()


def save_results(results_df, output_path="resultados_modelos.csv"):
    """
    Guarda los resultados de evaluación.
    """

    results_df.to_csv(
        output_path,
        index=False
    )

    print(f"Resultados guardados en: {output_path}")


def monitoring_summary(results_df):
    """
    Muestra un resumen de los resultados obtenidos.
    """

    print("\n===== RESUMEN DE MONITOREO =====")

    print(results_df)

    numeric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "PR-AUC"
    ]

    available_columns = [
        column
        for column in numeric_columns
        if column in results_df.columns
    ]

    if available_columns:
        print("\nPromedio de métricas:")
        print(results_df[available_columns].mean())


def monitor_predictions(predictions):
    """
    Monitorea las predicciones generadas por el modelo.
    """

    print("\n===== MONITOREO DE PREDICCIONES =====")

    total = len(predictions)

    print(f"\nTotal de predicciones: {total}")

    print("\nDistribución de predicciones:")

    distribution = predictions[
        "Clasificacion"
    ].value_counts()

    print(distribution)

    print("\nPorcentaje de predicciones:")

    percentages = (
        predictions["Clasificacion"]
        .value_counts(normalize=True) * 100
    )

    print(percentages.round(2))

    print("\nEstadísticas de probabilidad de incumplimiento:")

    print(
        predictions[
            "Probabilidad_incumplimiento"
        ].describe().round(4)
    )

    print("\nValores faltantes en las predicciones:")

    print(predictions.isnull().sum())

    # Gráfico de distribución de predicciones
    distribution.plot(
        kind="bar",
        figsize=(8, 5)
    )

    plt.title("Distribución de predicciones")
    plt.xlabel("Clasificación")
    plt.ylabel("Cantidad")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Gráfico de probabilidades
    predictions[
        "Probabilidad_incumplimiento"
    ].hist(
        bins=20,
        figsize=(8, 5)
    )

    plt.title("Distribución de probabilidades de incumplimiento")
    plt.xlabel("Probabilidad")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()


def run_monitoring():
    """
    Ejecuta el monitoreo del modelo desplegado.
    """

    predictions = deploy_model()

    monitor_predictions(predictions)

    return predictions


if __name__ == "__main__":

    run_monitoring()
