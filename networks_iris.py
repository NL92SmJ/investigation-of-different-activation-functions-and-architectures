# Module importieren
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ordner für die komplette Evaluation erstellen
os.makedirs("full_evaluation_plots_iris", exist_ok=True)

# Daten einlesen & Vorverarbeitung (einmalig)
path = "../Data/iris.csv"
data = pd.read_csv(path, delimiter=',')
data = data.drop(['sepal.width'], axis=1)
col = data['sepal.length']
data = data.drop(['sepal.length'], axis=1)

conv_ohe = ['species']
data = pd.get_dummies(data, columns=conv_ohe, dtype=float)

s_scaler = StandardScaler()
cols_to_s_scale = ['petal.length', 'petal.width']
data[cols_to_s_scale] = s_scaler.fit_transform(data[cols_to_s_scale])

# Train-Test-Split
train_data, test_data, train_col, test_col = train_test_split(data, col, test_size=0.2, random_state=42)

# --- 1. ARCHITEKTUREN ---
architectures = {
    "01_Standard_Ref": [32, 64], 
    "02_Klein": [32],
    "03_Breit": [64, 128],
    "04_Tief": [32, 64, 32],
    "05_Symmetrisch": [64, 64],
    "06_Pyramidal": [64, 32, 16],
    "07_Inverser_Trichter": [16, 32, 64],
    "08_Sanduhr_Bottleneck": [64, 16, 64]
}

# --- 2. AKTIVIERUNGSFUNKTIONEN ---
activations = [
    ("ReLU", "relu"),
    ("ELU", "elu"),
    ("GELU", "gelu"),
    ("Swish", "swish"),
    ("Sigmoid", "sigmoid"),
    ("SeLU", "selu"),
    ("Tanh", "tanh"),
    ("Leaky_ReLU", lambda: tf.keras.layers.LeakyReLU(alpha=0.1))
]
# Liste zum Sammeln aller Metriken für die Tabelle
summary_results = []

# --- 4. GESCHACHTELTE SCHLEIFE (GRID SEARCH) ---
for arch_name, layers_config in architectures.items():
    for act_name, act in activations:
        run_name = f"{arch_name}_{act_name}"
        print(f"Liefere ab: Architektur [{arch_name}] mit Aktivierung [{act_name}]...")
        
        # Modell dynamisch aufbauen
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(data.shape[1],)))
        
        for units in layers_config:
            if callable(act):
                model.add(tf.keras.layers.Dense(units))
                model.add(act())
            else:
                model.add(tf.keras.layers.Dense(units, activation=act))
                
        # Output-Layer für Regression
        model.add(tf.keras.layers.Dense(1))

        # Kompilieren
        model.compile(optimizer='adam', loss='mae', metrics=['mae'])

        # Training (70 Epochen)
        history = model.fit(
            train_data, train_col,
            epochs=70,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )

        # Vorhersagen, Statistiken & Metriken
        predictions = model.predict(test_data).flatten()
        mae = mean_absolute_error(test_col, predictions)
        r2 = r2_score(test_col, predictions)
        errors = predictions - test_col.values
        pred_std = np.std(predictions)
        error_std = np.std(errors)
        mean_pred = np.mean(predictions)

        # Ergebnisse für die Tabelle merken
        summary_results.append({
            "Modell": run_name,
            "Architektur": arch_name,
            "Aktivierung": act_name,
            "MAE": round(mae, 4),
            "R2_Score": round(r2, 4),
            "Pred_Mean": round(mean_pred, 4),
            "Pred_Std": round(pred_std, 4),
            "Error_Std": round(error_std, 4)
        })

        # --- PLOT 1: Loss-Verlauf ---
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Loss Verlauf ({run_name})')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.savefig(f"full_evaluation_plots_iris/loss_{run_name.lower()}.png")
        plt.close()

        # --- PLOT 2: Prediction vs Reality (MIT SEABORN JITTER) ---
        plt.figure(figsize=(6, 6))
        sns.regplot(
            x=test_col, 
            y=predictions, 
            x_jitter=0.15, 
            y_jitter=0.05,
            scatter_kws={'alpha': 0.4, 'color': 'blue'}, 
            line_kws={'color': 'red', 'linestyle': '--'}
        )
        plt.title(f"Prediction vs Reality (Jitter) - {run_name}")
        plt.xlabel("Echte Werte (Quality)")
        plt.ylabel("Vorhersage")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(f"full_evaluation_plots_iris/pred_vs_reality_{run_name.lower()}.png")
        plt.close()

        # --- PLOT 3: Fehlerverteilung (Histogramm) ---
        plt.figure(figsize=(8, 4))
        plt.hist(errors, bins=20, edgecolor='black', color='orange', alpha=0.7)
        plt.title(f"Fehlerverteilung ({run_name})")
        plt.xlabel("Vorhersagefehler")
        plt.ylabel("Anzahl")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(f"full_evaluation_plots_iris/error_hist_{run_name.lower()}.png")
        plt.close()

        # --- PLOT 4: Residual Plot ---
        plt.figure(figsize=(8, 4))
        plt.scatter(predictions, errors, alpha=0.5, color='purple')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel("Vorhersage")
        plt.ylabel("Fehler (Pred - Reality)")
        plt.title(f"Residual Plot ({run_name})")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(f"full_evaluation_plots_iris/residual_{run_name.lower()}.png")
        plt.close()

# --- 5. TABELLE ERSTELLEN UND AUSGEBEN ---
df_summary = pd.DataFrame(summary_results)

# Nach MAE sortieren (bester Wert oben)
df_summary = df_summary.sort_values(by="MAE", ascending=True).reset_index(drop=True)

print("\n" + "="*80)
print("GESAMT-AUSWERTUNG TABELLE (Sortiert nach bestem MAE)")
print("="*80)
print(df_summary.to_string(index=False))
print("="*80)

# Tabelle als CSV speichern
df_summary.to_csv("full_evaluation_summary_iris.csv", index=False)
print("\nTabelle wurde erfolgreich als 'full_evaluation_summary_iris.csv' gespeichert!")