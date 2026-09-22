import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def prepare_data(df):
    """
    Limpieza y preparación inicial de los datos.
    """

    df = df.copy()

    # ---------------------------------------------------
    # 1. Convertir fecha
    # ---------------------------------------------------
    if "fecha_prestamo" in df.columns:

        df["fecha_prestamo"] = pd.to_datetime(
            df["fecha_prestamo"],
            errors="coerce"
        )

        df["anio_prestamo"] = df["fecha_prestamo"].dt.year
        df["mes_prestamo"] = df["fecha_prestamo"].dt.month
        df["dia_semana_prestamo"] = df["fecha_prestamo"].dt.dayofweek

        df.drop(columns=["fecha_prestamo"], inplace=True)

    # ---------------------------------------------------
    # 2. Convertir puntaje a numérico
    # ---------------------------------------------------
    if "puntaje" in df.columns:

        df["puntaje"] = (
            df["puntaje"]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )

        df["puntaje"] = pd.to_numeric(
            df["puntaje"],
            errors="coerce"
        )

    # ---------------------------------------------------
    # 3. Limpiar tendencia_ingresos
    # ---------------------------------------------------
    if "tendencia_ingresos" in df.columns:

        valores_validos = [
            "Creciente",
            "Estable",
            "Decreciente"
        ]

        df["tendencia_ingresos"] = df[
            "tendencia_ingresos"
        ].where(
            df["tendencia_ingresos"].isin(valores_validos),
            pd.NA
        )

    return df


def build_preprocessor(X):
    """
    Construye el preprocesador para variables
    numéricas, categóricas y ordinales.
    """

    numeric_features = [
        col for col in [
            "capital_prestado",
            "plazo_meses",
            "edad_cliente",
            "salario_cliente",
            "total_otros_prestamos",
            "cuota_pactada",
            "puntaje",
            "puntaje_datacredito",
            "cant_creditosvigentes",
            "saldo_mora",
            "saldo_total",
            "saldo_principal",
            "saldo_mora_codeudor",
            "creditos_sectorFinanciero",
            "creditos_sectorCooperativo",
            "creditos_sectorReal",
            "promedio_ingresos_datacredito",
            "anio_prestamo",
            "mes_prestamo",
            "dia_semana_prestamo"
        ]
        if col in X.columns
    ]

    categorical_features = [
        col for col in [
            "tipo_credito",
            "tipo_laboral",
            "huella_consulta"
        ]
        if col in X.columns
    ]

    ordinal_features = [
        col for col in [
            "tendencia_ingresos"
        ]
        if col in X.columns
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            categories=[
                ["Decreciente", "Estable", "Creciente"]
            ],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    transformers = []

    if numeric_features:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    if ordinal_features:
        transformers.append(
            ("ordinal", ordinal_pipeline, ordinal_features)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


def prepare_train_test(df, target="Pago_atiempo"):
    """
    Prepara X e y y realiza la división
    entre entrenamiento y evaluación.
    """

    df = prepare_data(df)

    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    return X_train, X_test, y_train, y_test, preprocessor