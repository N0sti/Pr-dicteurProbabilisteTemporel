import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Charger les données
data = pd.read_json("donnees_historiques.json")

# Assurez-vous que les données ont un horodatage et sont bien triées
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.sort_values('timestamp')
data.set_index('timestamp', inplace=True)

# Extraire les températures et travailler sur les deux dernières années
temperature_series = data['temperature']
temperature_last_two_years = temperature_series.loc[data.index[-2 * 365 * 24:]]

# Identifier les paramètres SARIMA : ces valeurs peuvent être ajustées
p, d, q = 1, 1, 1
P, D, Q, m = 1, 1, 1, 24  # m=24 car les données sont horaires avec saisonnalité quotidienne

# Ajuster le modèle SARIMA
model = SARIMAX(temperature_last_two_years, 
                order=(p, d, q), 
                seasonal_order=(P, D, Q, m),
                enforce_stationarity=False, 
                enforce_invertibility=False)
results = model.fit()

# Prédictions pour les deux dernières années
start_date = temperature_last_two_years.index[0]
end_date = temperature_last_two_years.index[-1]
predictions = results.predict(start=start_date, end=end_date)

# Prédictions pour les 3 prochains jours (72 heures)
forecast = results.get_forecast(steps=72)
forecast_mean = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

# Évaluer la précision sur les deux dernières années
rmse = np.sqrt(mean_squared_error(temperature_last_two_years, predictions))
print(f"RMSE sur les deux dernières années : {rmse:.2f}")

# Tracer les résultats
plt.figure(figsize=(12, 6))

# Données observées
plt.plot(temperature_last_two_years, label="Données observées", color='blue')

# Prédictions SARIMA
plt.plot(predictions, label="Prédictions (2 dernières années)", color='orange')

# Prévisions futures (3 jours)
future_dates = pd.date_range(start=temperature_last_two_years.index[-1], periods=73, freq='H')[1:]
plt.plot(future_dates, forecast_mean, label="Prédictions futures (3 jours)", color='green')

# Intervalle de confiance
plt.fill_between(future_dates, 
                 confidence_intervals.iloc[:, 0], 
                 confidence_intervals.iloc[:, 1], 
                 color='green', alpha=0.2, label="Intervalle de confiance")

# Mise en forme
plt.title("Prédictions de température avec SARIMA")
plt.xlabel("Temps")
plt.ylabel("Température (°C)")
plt.legend()
plt.show()
