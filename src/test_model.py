import pandas as pd

from model_training_evaluation import train_and_evaluate
from model_monitoring import compare_models, save_results


df = pd.read_csv("Base_de_datos1.csv")

results_df, trained_models, X_train, X_test, y_train, y_test = train_and_evaluate(
    df,
    target="Pago_atiempo"
)

print("\n===== RESULTADOS =====")
print(results_df)

compare_models(results_df)

save_results(
    results_df,
    "resultados_modelos.csv"
)

