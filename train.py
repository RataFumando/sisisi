import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

# === 1. Cargar datos ===
df = pd.read_csv("data.csv")

df.dropna(inplace=True)

# === 2. Separar características y etiqueta ===
X = df.drop(["maxtemp"], axis=1)  # Predecimos maxtemp
y = df["maxtemp"]

# Columnas categóricas
categorical_columns = ["weather", "cloud", "Date"]

# Asegurar que la fecha se maneje como categoría sin procesar
X["Date"] = X["Date"].astype(str)

# === 3. Preprocesamiento ===
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ],
    remainder="passthrough"
)

# === 4. Modelo Random Forest ===
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Pipeline completo
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# === 5. Entrenar modelo ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

# === 6. Evaluación ===
y_pred = pipeline.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# === 7. Extraer importancias del Random Forest ===
# NOTA: Debemos obtener los nombres de las columnas luego del OneHotEncoder
encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
encoded_cols = encoder.get_feature_names_out(categorical_columns)

# Columnas numéricas (las que no son categóricas)
numeric_cols = [col for col in X.columns if col not in categorical_columns]

# Combinar nombres finales
feature_names = list(encoded_cols) + numeric_cols

importances = pipeline.named_steps["model"].feature_importances_

# Ordenar por importancia
indices = np.argsort(importances)[::-1]

# === 8. Graficar ===
plt.figure(figsize=(12, 6))
plt.title("Importancia de Características - Random Forest")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# === 9. Guardar modelo ===
joblib.dump(pipeline, "random_forest_model.pkl")
print("Modelo guardado como random_forest_model.pkl")
