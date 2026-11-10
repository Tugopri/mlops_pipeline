
import pandas as pd
from src.model_training_evaluation import train_and_evaluate
from src.model_monitoring import compare_models, save_results

def test_model():

    print("\n===== INICIANDO PRUEBAS DEL MODELO =====")

    # Cargar los datos
    df = pd.read_csv("Base_de_datos1.csv")

    print("\nDatos cargados correctamente.")
    print("Dimensiones del dataset:", df.shape)

    # Entrenar y evaluar los modelos
    results_df, trained_models, X_train, X_test, y_train, y_test = train_and_evaluate(
        df,
        target="Pago_atiempo"
    )

    # Mostrar resultados
    print("\n===== RESULTADOS DE LOS MODELOS =====")
    print(results_df.to_string(index=False))

    # Verificar resultados
    assert not results_df.empty, "No se generaron resultados."

    assert len(trained_models) > 0, "No se entrenaron modelos."

    assert len(X_test) == len(y_test), "Los datos de prueba no coinciden."

    print("\nPruebas básicas completadas correctamente.")

    # Comparar modelos gráficamente
    compare_models(results_df)

    # Guardar resultados
    save_results(
        results_df,
        "resultados_modelos.csv"
    )

    print("\n===== PRUEBAS FINALIZADAS =====")

    return results_df, trained_models


if __name__ == "__main__":
    test_model()
