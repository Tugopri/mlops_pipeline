import pandas as pd
import matplotlib.pyplot as plt


def compare_models(results_df):
    """
    Genera una comparación gráfica de las métricas
    obtenidas por los modelos.
    """

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
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

    ax.set_title(
        "Comparación de modelos de clasificación"
    )

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

    print(
        f"Resultados guardados en: {output_path}"
    )


def monitoring_summary(results_df):
    """
    Muestra un resumen de los resultados obtenidos.
    """

    print("===== RESUMEN DE MONITOREO =====")
    print()

    print(results_df)

    print()

    print("Promedio de métricas:")

    numeric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]

    available_columns = [
        column
        for column in numeric_columns
        if column in results_df.columns
    ]

    print(
        results_df[available_columns].mean()
    )