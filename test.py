from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import json

# Fonction pour charger les données historiques
def charger_donnees_historiques():
    try:
        with open('donnees_historiques.json', 'r') as fichier:
            donnees = json.load(fichier)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Aucune donnée valide trouvée pour afficher le graphique.")
        return None
    return donnees

# Fonction pour préparer les données pour Prophet
def preparer_donnees_prophet(donnees_historiques):
    timestamps = donnees_historiques[0]['hourly']['time']
    temperatures = donnees_historiques[0]['hourly']['temperature_2m']
    df = pd.DataFrame({'ds': pd.to_datetime(timestamps), 'y': temperatures})
    return df

# Fonction pour entraîner le modèle Prophet et faire des prédictions
def entrainer_et_predire_prophet(df, future_steps=3*365*24):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=future_steps, freq='h')
    forecast = model.predict(future)
    return forecast

# Fonction pour afficher les résultats
def afficher_resultats_prophet(df, forecast):
    plt.figure(figsize=(12, 7))
    plt.plot(df['ds'], df['y'], label='Données observées')
    plt.plot(forecast['ds'], forecast['yhat'], label='Prédictions', color='orange')
    plt.xlabel('Temps')
    plt.ylabel('Température (°C)')
    plt.title('Prédictions de température pour les trois prochaines années')
    plt.legend()
    plt.show()

# Programme principal
if __name__ == "__main__":
    donnees_historiques = charger_donnees_historiques()
    if donnees_historiques is None:
        print("Aucune donnée historique trouvée. Arrêt du programme.")
        exit(1)

    df = preparer_donnees_prophet(donnees_historiques)
    forecast = entrainer_et_predire_prophet(df)
    afficher_resultats_prophet(df, forecast)
