
import pandas as pd
import joblib

from pathlib import Path

from src.ft_engineering import prepare_data


# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# Modelo seleccionado inicialmente
MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"


def load_model(model_path=MODEL_PATH):
    """
    Carga el modelo previamente entrenado.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {model_path}. "
            "Ejecuta primero el entrenamiento."
        )

    model = joblib.load(model_path)

    print(f"Modelo cargado correctamente: {model_path.name}")

    return model


def prepare_new_data(df):
    """
    Prepara nuevos registros utilizando las mismas
    transformaciones iniciales del entrenamiento.
    """

    df = prepare_data(df)

    # Eliminar target si está presente
    if "Pago_atiempo" in df.columns:
        df = df.drop(columns=["Pago_atiempo"])

    # Excluir variable descartada por posible fuga de información
    if "puntaje" in df.columns:
        df = df.drop(columns=["puntaje"])

    return df


def predict_new_data(df, model=None):
    """
    Realiza predicciones sobre nuevos préstamos.
    """

    if model is None:
        model = load_model()

    X_new = prepare_new_data(df)

    predictions = model.predict(X_new)

    class_zero_index = list(model.classes_).index(0)

    probabilities = model.predict_proba(X_new)[
        :, class_zero_index
    ]

    results = pd.DataFrame({
        "Prediccion_Pago_atiempo": predictions,
        "Probabilidad_incumplimiento": probabilities
    }, index=df.index)

    results["Clasificacion"] = results[
        "Prediccion_Pago_atiempo"
    ].map({
        0: "Posible incumplimiento",
        1: "Posible cumplimiento"
    })

    return results


def deploy_model(data_path=None):
    """
    Ejecuta el proceso de despliegue y predicción.
    """

    if data_path is None:
        data_path = BASE_DIR / "Base_de_datos1.csv"

    df = pd.read_csv(data_path)

    model = load_model()

    predictions = predict_new_data(
        df,
        model=model
    )

    print("\nRESULTADOS DE PREDICCIÓN")
    print(predictions.head(10).to_string(index=False))

    print("\nRESUMEN DE PREDICCIONES")
    print(
        predictions["Clasificacion"].value_counts()
    )

    return predictions


if __name__ == "__main__":

    deploy_model()
