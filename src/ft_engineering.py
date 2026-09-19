import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split



def prepare_data(df):

    df = df.copy()

    if "fecha_prestamo" in df.columns:
        df["fecha_prestamo"] = pd.to_datetime(
            df["fecha_prestamo"],
            errors="coerce"
        )


def build_feature_pipeline():

    # ---------------------------------------------------
    # 2. Variables numéricas
    # ---------------------------------------------------
    numeric_features = [
        "capital_prestado",
        "plazo_meses",
        "edad_cliente",
        "salario_cliente",
        "total_otros_prestamos",
        "cuota_pactada",
        "puntaje_datacredito",
        "cant_creditosvigentes",
        "huella_consulta",
        "saldo_mora",
        "saldo_total",
        "saldo_principal",
        "saldo_mora_codeudor",
        "creditos_sectorFinanciero",
        "creditos_sectorCooperativo",
        "creditos_sectorReal",
        "anio_prestamo",
        "mes_prestamo",
        "dia_semana_prestamo"
    ]

    # ---------------------------------------------------
    # 3. Variables categóricas nominales
    # ---------------------------------------------------
    categorical_features = [
        "tipo_credito",
        "tipo_laboral"
    ]

    # ---------------------------------------------------
    # 4. Variables ordinales
    # ---------------------------------------------------
    ordinal_features = [
        "puntaje",
        "tendencia_ingresos"
    ]

    # Orden de las categorías:
    # ESTO DEBEMOS CONFIRMARLO CON TUS DATOS REALES
    ordinal_categories = [
        [
            "Bajo",
            "Medio",
            "Alto"
        ],
        [
            "Disminuye",
            "Estable",
            "Aumenta"
        ]
    ]

    # ---------------------------------------------------
    # 5. Pipeline numérico
    # ---------------------------------------------------
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            ("scaler",
             StandardScaler()

            )
        ]
    )

    # ---------------------------------------------------
    # 6. Pipeline categórico
    # ---------------------------------------------------
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # ---------------------------------------------------
    # 7. Pipeline ordinal
    # ---------------------------------------------------
    ordinal_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                )
            )
        ]
    )

    # ---------------------------------------------------
    # 8. Unir todos los pipelines
    # ---------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            ),
            (
                "ordinal",
                ordinal_pipeline,
                ordinal_features
            )
        ]
    )

    return preprocessor


def split_data(df, target_column="Pago_atiempo"):

    # ---------------------------------------------------
    # 9. Separar X e y
    # ---------------------------------------------------
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # ---------------------------------------------------
    # 10. Train / Test
    # ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test