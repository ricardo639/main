import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Cargar datos (ajusta la ruta según corresponda)
data = pd.read_csv("C:/Users/perez/OneDrive/Documentos/UNIVERSIDAD/SEMESTRES/SEPTIMO SEMESTRE/PROBABILIDAD Y ESTADISTICA/TRABAJO TECNICO/insurance_challenge_train.csv")

# Separar variables predictoras (X) y variable objetivo (y)
X = data.drop("APERSAUT", axis=1)  # APERSAUT como variable objetivo
y = data["APERSAUT"]

# Identificar columnas categóricas
categorical_columns = X.select_dtypes(include=['object']).columns

# Convertir columnas categóricas a numéricas usando LabelEncoder
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Convertir 'y' en valores numéricos si es categórica
if y.dtype == 'object':
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Construir y entrenar la red neuronal MLPClassifier (problema de clasificación)
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, verbose=True)
model.fit(X_train_scaled, y_train)

# Predicción y evaluación
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Obtener importancia de características mediante permutation_importance
importance = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42)
for i in importance.importances_mean.argsort()[::-1]:
    print(f"{X.columns[i]}: {importance.importances_mean[i]:.4f}")

# Gráfica del ajuste de la red neuronal (pérdida a lo largo de las iteraciones)
plt.plot(model.loss_curve_)
plt.title("Evolución de la pérdida durante el entrenamiento")
plt.xlabel("Iteraciones")
plt.ylabel("Pérdida (Loss)")
plt.grid(True)
plt.show()
