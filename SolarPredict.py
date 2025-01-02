from asyncio import sleep
import os
import subprocess
import requests
import json
import datetime
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import torch
import pandas as pd
from prophet import Prophet
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive


# Paramètres d'entrée
puissance_nominale_par_panneau = 300  # Wc
nombre_de_panneaux = 9
surface_par_panneau = 1.6  # m²
efficacite_panneaux = 0.20  # 20%
inclinaison_panneaux = 30  # degrés
orientation_panneaux = 0  # degrés par rapport au sud (0 degrés = plein sud)
facteur_de_performance = 0.85

# Fonction pour obtenir les données météorologiques actuelles
def obtenir_donnees_meteo_actuelles_hourly(current_datetime):
    # Formater start_date et end_date au format requis (YYYY-MM-DDTHH:MM)
    start_date = current_datetime.strftime('%Y-%m-%dT%H:%M')
    end_date = current_datetime.strftime('%Y-%m-%dT%H:%M')

    # Construire l'URL avec les dates dynamiques
    url = (f'https://api.open-meteo.com/v1/forecast?latitude=48.7833&longitude=2.3333&&start_hour={start_date}&end_hour={end_date}&hourly=temperature_2m,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high&timezone=Europe%2FBerlin')
    # Faire la requête HTTP
    print("url one is", url)
    response = requests.get(url)
    response.raise_for_status()  # Vérifie les erreurs HTTP
    data = response.json()
    
    return data

def obtenir_donnees_meteo_actuelles_daily(current_datetime):
    # Formater start_date et end_date au format requis (YYYY-MM-DDTHH:MM)
    start_date = current_datetime.strftime('%Y-%m-%d')
    end_date = current_datetime.strftime('%Y-%m-%d')

    # Construire l'URL avec les dates dynamiques
    url = (f'https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=48.7833&longitude=2.3333&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high&daily=sunrise,sunset,daylight_duration,sunshine_duration&timezone=Europe%2FBerlin')
    # Faire la requête HTTP
    print("url one is", url)
    response = requests.get(url)
    response.raise_for_status()  # Vérifie les erreurs HTTP
    data = response.json()
    #print("data", data)
    
    return data

# Fonction pour calculer l'ensoleillement et la température actuels
def calculer_ensoleillement_et_temperature(data):
    #print(data)  # Débogage pour afficher les données reçues
    
    # Extraire les données horaires
    hourly_data = data.get('hourly', {})
    if not hourly_data:
        raise KeyError("Les données 'hourly' sont manquantes dans la réponse API.")

    # Obtenir les valeurs pour la première (et unique) heure retournée
    temperatures = hourly_data.get('temperature_2m', [])
    cloud_covers = hourly_data.get('cloud_cover', [])
    
    if not temperatures or not cloud_covers:
        raise ValueError("Les données 'temperature_2m' ou 'cloud_cover' sont manquantes ou vides.")
    
    # Assumer que la première valeur correspond à l'heure actuelle
    temperature_actuelle = temperatures[0]
    cloud_cover = cloud_covers[0]

    # Calcul de l'ensoleillement (100% - couverture nuageuse)
    ensoleillement_actuel = (100 - cloud_cover) / 100  # Convertir en fraction
    print(f"Ensoleillement actuel: {ensoleillement_actuel * 100}%")
    print(f"Température actuelle: {temperature_actuelle}°C")
    return ensoleillement_actuel, temperature_actuelle

def mettre_a_jour_historique_hourly(data):
    #print("testststts", data)
    fichier_historique = 'donnees_historiques.json'
    # Charger le fichier historique
    with open(fichier_historique, 'r') as file:
        historique = json.load(file)
    #print("data", data)
    #print(data['hourly']['time'])
    #print(data['hourly']['temperature_2m'])
    #print(data['hourly']['cloud_cover'])
    #print(data['hourly']['cloud_cover_low'])
    #print(data['hourly']['cloud_cover_mid'])
    #print(data['hourly']['cloud_cover_high'])
    # Ajouter les nouvelles données à la section "hourly" de l'historique
    #print("historique",  historique[0]['hourly']['time'])
    for item in data['hourly']['time']:
        historique[0]['hourly']['time'].append(item)
    #print("historique",  historique[0]['hourly']['time'])
    for item in data['hourly']['temperature_2m']:
        historique[0]['hourly']['temperature_2m'].append(item)
    #print("historique",  historique[0]['hourly']['temperature_2m'])
    for item in data['hourly']['cloud_cover']:
        historique[0]['hourly']['cloud_cover'].append(item)
    #print("historique",  historique[0]['hourly']['cloud_cover'])
    for item in data['hourly']['cloud_cover_low']:
        historique[0]['hourly']['cloud_cover_low'].append(item)
    #print("historique",  historique[0]['hourly']['cloud_cover_low'])
    for item in data['hourly']['cloud_cover_mid']:
        historique[0]['hourly']['cloud_cover_mid'].append(item)
    #print("historique",  historique[0]['hourly']['cloud_cover_mid'])
    for item in data['hourly']['cloud_cover_high']:
        historique[0]['hourly']['cloud_cover_high'].append(item)
    #print("historique",  historique[0]['hourly']['cloud_cover_high'])
    # Sauvegarder les modifications dans le fichier historique
    with open(fichier_historique, 'w') as file:
        json.dump(historique, file, indent=4)
    print("Historique mis à jour avec succès.")

def mettre_a_jour_historique_daily(data):
    #print("testststts", data)
    fichier_historique = 'donnees_historiques.json'
    # Charger le fichier historique
    with open(fichier_historique, 'r') as file:
        historique = json.load(file)
    #print("data", data)
    #print(data['daily']['time'])
    #print(data['daily']['sunrise'])
    #print(data['daily']['sunset'])
    #print(data['daily']['daylight_duration'])
    #print(data['daily']['sunshine_duration'])
    # Ajouter les nouvelles données à la section "hourly" de l'historique
    #print("historique",  historique[0]['daily']['time'])
    for item in data['daily']['time']:
        historique[0]['daily']['time'].append(item)
    #print("historique",  historique[0]['daily']['time'])

    for item in data['daily']['sunrise']:
        historique[0]['daily']['sunrise'].append(item)
    #print("historique",  historique[0]['daily']['sunrise'])

    for item in data['daily']['sunset']:
        historique[0]['daily']['sunset'].append(item)
    #print("historique",  historique[0]['daily']['sunset'])

    for item in data['daily']['daylight_duration']:
        historique[0]['daily']['daylight_duration'].append(item)
    #print("historique",  historique[0]['daily']['daylight_duration'])

    for item in data['daily']['sunshine_duration']:
        historique[0]['daily']['sunshine_duration'].append(item)
    #print("historique",  historique[0]['daily']['sunshine_duration'])

    #Sauvegarder les modifications dans le fichier historique
    with open(fichier_historique, 'w') as file:
        json.dump(historique, file, indent=4)
    print("Historique mis à jour avec succès.")

# Fonction pour prédire la production d'électricité en temps réel
def predire_production_electricite(puissance_nominale_par_panneau, nombre_de_panneaux, surface_par_panneau,
                                   efficacite_panneaux, ensoleillement_actuel, inclinaison_panneaux,
                                   orientation_panneaux, facteur_de_performance, temperature_actuelle):
    puissance_installée = puissance_nominale_par_panneau * nombre_de_panneaux / 1000  # kWc
    surface_totale = surface_par_panneau * nombre_de_panneaux  # m²

    # Ajustement de l'efficacité en fonction de la température
    temperature_coefficient = -0.004  # Coefficient de température typique pour les panneaux solaires
    temperature_standard = 25  # Température standard de test (STC)
    efficacite_ajustee = efficacite_panneaux * (1 + temperature_coefficient * (temperature_actuelle - temperature_standard))

    energie_produite_heure = puissance_installée * ensoleillement_actuel * efficacite_ajustee * facteur_de_performance  # kWh
    return energie_produite_heure

#but aller chopper les donne d'entrainement pour pourvoir faire des prediction
def donnee_entrainement(donnees_historiques):
    hourly_timestamps_entrainement = donnees_historiques[0]['hourly']['time']
    daily_timestamps_entrainement = donnees_historiques[0]['daily']['time']
    temperatures_entrainement = donnees_historiques[0]['hourly']['temperature_2m']
    cloud_covers_entrainement = donnees_historiques[0]['hourly']['cloud_cover']
    sunrise_entrainement = donnees_historiques[0]['daily']['sunrise']
    sunset_entrainement = donnees_historiques[0]['daily']['sunset']

    temperatures_tenseur = torch.tensor(temperatures_entrainement, dtype=torch.float32)
    cloud_covers_tenseur = torch.tensor(cloud_covers_entrainement, dtype=torch.float32)
    sunrise_tenseur = torch.tensor([datetime.strptime(time, '%Y-%m-%dT%H:%M').hour + datetime.strptime(time, '%Y-%m-%dT%H:%M').minute / 60 for time in sunrise_entrainement], dtype=torch.float32)
    sunset_tenseur = torch.tensor([datetime.strptime(time, '%Y-%m-%dT%H:%M').hour + datetime.strptime(time, '%Y-%m-%dT%H:%M').minute / 60 for time in sunset_entrainement], dtype=torch.float32)

    time_temperatures = torch.arange(len(temperatures_tenseur), dtype=torch.float32)
    time_sunrise = torch.arange(len(sunrise_tenseur), dtype=torch.float32)
    time_sunset = torch.arange(len(sunset_tenseur), dtype=torch.float32)

    print(f"Shape of temperatures: {temperatures_tenseur.shape}")
    print(f"Shape of cloud_covers: {cloud_covers_tenseur.shape}")
    print(f"Shape of sunrise: {sunrise_tenseur.shape}")
    print(f"Shape of sunset: {sunset_tenseur.shape}")
    print(f"Shape of time_temperatures: {time_temperatures.shape}")
    print(f"Shape of time_sunrise: {time_sunrise.shape}")
    print(f"Shape of time_sunset: {time_sunset.shape}")

    return temperatures_tenseur, cloud_covers_tenseur, sunrise_tenseur, sunset_tenseur, time_temperatures, time_sunrise, time_sunset

def model_sunrise_sinusoidal(time, sunrise):
    a = pyro.sample("a", dist.Normal(0., 10.))
    b = pyro.sample("b", dist.Normal(0., 10.))
    c = pyro.sample("c", dist.Normal(0., 10.))
    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))

    mean = a + b * torch.sin(2 * np.pi * time / 365) + c * torch.cos(2 * np.pi * time / 365)

    if sunrise is not None:
        with pyro.plate("data", len(sunrise)):
            return pyro.sample("obs", dist.Normal(mean[:len(sunrise)], sigma), obs=sunrise)
    else:
        with pyro.plate("data", len(time)):
            return pyro.sample("obs", dist.Normal(mean, sigma))

def model_sunset_sinusoidal(time, sunset):
    a = pyro.sample("a", dist.Normal(0., 10.))
    b = pyro.sample("b", dist.Normal(0., 10.))
    c = pyro.sample("c", dist.Normal(0., 10.))
    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))

    mean = a + b * torch.sin(2 * np.pi * time / 365) + c * torch.cos(2 * np.pi * time / 365)

    if sunset is not None:
        with pyro.plate("data", len(sunset)):
            return pyro.sample("obs", dist.Normal(mean[:len(sunset)], sigma), obs=sunset)
    else:
        with pyro.plate("data", len(time)):
            return pyro.sample("obs", dist.Normal(mean, sigma))

def predict_next_three_days_sunrise(time_sunrise, sunrise, current_datetime):
    """
    Predict sunrise times for the next three days using the existing sinusoidal model
    """
    # Effectuer l'inférence avec le modèle existant
    nuts_kernel = NUTS(model_sunrise_sinusoidal)
    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
    mcmc.run(time_sunrise, sunrise)
    samples = mcmc.get_samples()

    # Créer les points temporels pour les trois prochains jours
    # Comme le modèle utilise une période de 365 jours, nous devons ajuster les valeurs en conséquence
    current_day_of_year = current_datetime.timetuple().tm_yday
    next_three_days = torch.tensor([
        current_day_of_year + 1,
        current_day_of_year + 2,
        current_day_of_year + 3
    ], dtype=torch.float32)

    # Faire les prédictions
    predictive = Predictive(model_sunrise_sinusoidal, samples)
    predictions = predictive(next_three_days, None)

    # Convertir les prédictions en heures et minutes
    prediction_dates = []
    prediction_times = []

    mean_predictions = predictions["obs"].mean(axis=0)
    std_predictions = predictions["obs"].std(axis=0)

    for i in range(3):
        next_date = current_datetime + timedelta(days=i+1)
        # La prédiction donne l'heure en format décimal (ex: 6.5 pour 6h30)
        predicted_hour = int(mean_predictions[i].item())
        predicted_minute = int((mean_predictions[i].item() % 1) * 60)

        prediction_dates.append(next_date.strftime('%Y-%m-%d'))
        prediction_times.append(f"{predicted_hour:02d}:{predicted_minute:02d}")

    return prediction_dates, prediction_times, mean_predictions, std_predictions

def predict_and_display_sunrise(time_sunrise, sunrise, current_datetime):
    """
    Predict sunrise times and ensure consistent display between graph and console
    """
    # Effectuer l'inférence et les prédictions comme avant
    nuts_kernel = NUTS(model_sunrise_sinusoidal)
    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
    mcmc.run(time_sunrise, sunrise)
    samples = mcmc.get_samples()

    # Créer les points temporels pour le graphique (données historiques + 3 jours)
    future_steps = 3
    full_time = torch.arange(len(sunrise) + future_steps, dtype=torch.float32)

    # Faire les prédictions pour tout l'intervalle
    predictive = Predictive(model_sunrise_sinusoidal, samples)
    predictions = predictive(full_time, None)

    # Afficher le graphique avec toutes les données
    plt.figure(figsize=(12, 6))
    plt.plot(time_sunrise.numpy(), sunrise.numpy(), 'o', label='Données observées', alpha=0.5)

    mean_predictions = predictions["obs"].mean(axis=0)
    std_predictions = predictions["obs"].std(axis=0)

    plt.plot(full_time.numpy(), mean_predictions.numpy(), 'r-', label='Prédictions')
    plt.fill_between(
        full_time.numpy(),
        (mean_predictions - std_predictions).numpy(),
        (mean_predictions + std_predictions).numpy(),
        color='r', alpha=0.2, label='Intervalle de confiance'
    )

    # Configure y-axis (time of day)
    hours = np.arange(4, 23, 0.25)  # From 4:00 to 22:00 with 15-min intervals
    plt.yticks(hours, [f"{int(h):02d}:{int((h % 1) * 60):02d}" for h in hours])

    # Configure x-axis with month labels and daily grid
    num_points = len(sunrise) + future_steps

    # Create month labels at regular intervals
    x_ticks_months = np.linspace(0, num_points-1, 6)  # Show 6 month labels
    current_date = current_datetime
    start_date = current_date - timedelta(days=len(sunrise))
    x_labels = [(start_date + timedelta(days=int(x))).strftime('%m/%Y') for x in x_ticks_months]

    # Set monthly labels
    plt.xticks(x_ticks_months, x_labels, rotation=45)

    # Create daily grid lines
    x_ticks_days = np.arange(0, num_points, 1)  # One tick per day

    # Add grid
    # Major grid for hours (horizontal lines)
    plt.grid(True, which='major', axis='y', linestyle='-', alpha=0.3)

    # Add vertical lines for each day
    for x in x_ticks_days:
        plt.axvline(x=x, color='gray', linestyle='-', alpha=0.1)

    # Add finer grid for 15-minute intervals
    plt.grid(True, which='minor', axis='y', linestyle=':', alpha=0.2)

    # Labels and title
    plt.xlabel('Mois')
    plt.ylabel('Heure du lever du soleil')
    plt.title('Prédictions du lever du soleil')

    # Adjust layout and legend
    plt.legend(loc='best')
    plt.tight_layout()

    #plt.show()

    # Extraire les 3 dernières prédictions (3 prochains jours)
    future_predictions = mean_predictions[-future_steps:]
    future_std = std_predictions[-future_steps:]

    # Préparer l'affichage console
    prediction_results = []
    for i in range(future_steps):
        next_date = current_datetime + timedelta(days=i+1)
        predicted_hour = int(future_predictions[i].item())
        predicted_minute = int((future_predictions[i].item() % 1) * 60)
        confidence_minutes = int(future_std[i].item() * 60)

        prediction_results.append({
            'date': next_date.strftime('%Y-%m-%d'),
            'time': f"{predicted_hour:02d}:{predicted_minute:02d}",
            'confidence': confidence_minutes
        })

    return prediction_results

def predict_next_three_days_sunset(time_sunset, sunset, current_datetime):
    """
    Predict sunset times for the next three days using the existing sinusoidal model
    """
    # Effectuer l'inférence avec le modèle existant
    nuts_kernel = NUTS(model_sunset_sinusoidal)
    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
    mcmc.run(time_sunset, sunset)
    samples = mcmc.get_samples()

    # Créer les points temporels pour les trois prochains jours
    # Comme le modèle utilise une période de 365 jours, nous devons ajuster les valeurs en conséquence
    current_day_of_year = current_datetime.timetuple().tm_yday
    next_three_days = torch.tensor([
        current_day_of_year + 1,
        current_day_of_year + 2,
        current_day_of_year + 3
    ], dtype=torch.float32)

    # Faire les prédictions
    predictive = Predictive(model_sunset_sinusoidal, samples)
    predictions = predictive(next_three_days, None)

    # Convertir les prédictions en heures et minutes
    prediction_dates = []
    prediction_times = []

    mean_predictions = predictions["obs"].mean(axis=0)
    std_predictions = predictions["obs"].std(axis=0)

    for i in range(3):
        next_date = current_datetime + timedelta(days=i+1)
        # La prédiction donne l'heure en format décimal (ex: 6.5 pour 6h30)
        predicted_hour = int(mean_predictions[i].item())
        predicted_minute = int((mean_predictions[i].item() % 1) * 60)

        prediction_dates.append(next_date.strftime('%Y-%m-%d'))
        prediction_times.append(f"{predicted_hour:02d}:{predicted_minute:02d}")

    return prediction_dates, prediction_times, mean_predictions, std_predictions

def predict_and_display_sunset(time_sunset, sunset, current_datetime):
    """
    Predict sunset times and ensure consistent display between graph and console
    """
    # Effectuer l'inférence et les prédictions comme avant
    nuts_kernel = NUTS(model_sunset_sinusoidal)
    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
    mcmc.run(time_sunset, sunset)
    samples = mcmc.get_samples()

    # Créer les points temporels pour le graphique (données historiques + 3 jours)
    future_steps = 3
    full_time = torch.arange(len(sunset) + future_steps, dtype=torch.float32)

    # Faire les prédictions pour tout l'intervalle
    predictive = Predictive(model_sunset_sinusoidal, samples)
    predictions = predictive(full_time, None)

    # Afficher le graphique avec toutes les données
    plt.figure(figsize=(12, 6))
    plt.plot(time_sunset.numpy(), sunset.numpy(), 'o', label='Données observées', alpha=0.5)

    mean_predictions = predictions["obs"].mean(axis=0)
    std_predictions = predictions["obs"].std(axis=0)

    plt.plot(full_time.numpy(), mean_predictions.numpy(), 'r-', label='Prédictions')
    plt.fill_between(
        full_time.numpy(),
        (mean_predictions - std_predictions).numpy(),
        (mean_predictions + std_predictions).numpy(),
        color='r', alpha=0.2, label='Intervalle de confiance'
    )

    # Configure y-axis (time of day)
    hours = np.arange(4, 23, 0.25)  # From 4:00 to 22:00 with 15-min intervals
    plt.yticks(hours, [f"{int(h):02d}:{int((h % 1) * 60):02d}" for h in hours])

    # Configure x-axis with month labels and daily grid
    num_points = len(sunset) + future_steps

    # Create month labels at regular intervals
    x_ticks_months = np.linspace(0, num_points-1, 6)  # Show 6 month labels
    current_date = current_datetime
    start_date = current_date - timedelta(days=len(sunset))
    x_labels = [(start_date + timedelta(days=int(x))).strftime('%m/%Y') for x in x_ticks_months]

    # Set monthly labels
    plt.xticks(x_ticks_months, x_labels, rotation=45)

    # Create daily grid lines
    x_ticks_days = np.arange(0, num_points, 1)  # One tick per day

    # Add grid
    # Major grid for hours (horizontal lines)
    plt.grid(True, which='major', axis='y', linestyle='-', alpha=0.3)

    # Add vertical lines for each day
    for x in x_ticks_days:
        plt.axvline(x=x, color='gray', linestyle='-', alpha=0.1)

    # Add finer grid for 15-minute intervals
    plt.grid(True, which='minor', axis='y', linestyle=':', alpha=0.2)

    # Labels and title
    plt.xlabel('Mois')
    plt.ylabel('Heure du coucher du soleil')
    plt.title('Prédictions du coucher du soleil')

    # Adjust layout and legend
    plt.legend(loc='best')
    plt.tight_layout()

    #plt.show()

    # Extraire les 3 dernières prédictions (3 prochains jours)
    future_predictions = mean_predictions[-future_steps:]
    future_std = std_predictions[-future_steps:]

    # Préparer l'affichage console
    prediction_results = []
    for i in range(future_steps):
        next_date = current_datetime + timedelta(days=i+1)
        predicted_hour = int(future_predictions[i].item())
        predicted_minute = int((future_predictions[i].item() % 1) * 60)
        confidence_minutes = int(future_std[i].item() * 60)

        prediction_results.append({
            'date': next_date.strftime('%Y-%m-%d'),
            'time': f"{predicted_hour:02d}:{predicted_minute:02d}",
            'confidence': confidence_minutes
        })

    return prediction_results

def charger_donnees_historiques():
    try:
        with open('donnees_historiques.json', 'r') as fichier:
            donnees = json.load(fichier)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Aucune donnée valide trouvée pour afficher le graphique.")
        return None
    return donnees

# Fonction pour stocker les nouvelles valeurs dans un fichier JSON dans le fichier historique
def stocker_donnees_json(energie_produite, current_datetime ):
    donnees = {
        "energie_produite": energie_produite,
        "timestamp": current_datetime.strftime('%d/%m/%Y %H:%M:%S')
    }

    try:
        with open('donnees_graphs.json', 'r') as f:
            donnees_existantes = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        donnees_existantes = []

    donnees_existantes.append(donnees)

    with open('donnees_graphs.json', 'w') as f:
        json.dump(donnees_existantes, f, indent=4)

    #print(f"Nouvelle donnée ajoutée: {donnees}")

# Fonction pour afficher un graphique de la production d'électricité
def afficher_graphique(predictions_future=None):
    file_path="graphique.png"
    try:
        with open('donnees_graphs.json', 'r') as fichier:
            donnees = json.load(fichier)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Aucune donnée valide trouvée pour afficher le graphique.")
        return

    if not donnees:
        print("Le fichier JSON ne contient aucune donnée.")
        return

    # Historical data
    timestamps = []
    energie_produite = []
    
    for entry in donnees:
        if 'timestamp' in entry and 'energie_produite' in entry:
            timestamps.append(datetime.strptime(entry['timestamp'], '%d/%m/%Y %H:%M:%S'))
            energie_produite.append(entry['energie_produite'])

    # Future predictions
    future_timestamps = []
    future_energie = []
    
    if predictions_future:
        for timestamp_str, energy in predictions_future:
            future_timestamps.append(datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S'))
            future_energie.append(energy)

    plt.figure(figsize=(12, 7))
    
    # Plot historical data
    plt.plot(timestamps, energie_produite, marker='o', linestyle='-', color='blue', 
             label='Production historique', markersize=4)
    
    # Plot future predictions
    if predictions_future:
        plt.plot(future_timestamps, future_energie, marker='o', linestyle='--', color='red',
                label='Prévisions', markersize=4)

    plt.xlabel('Temps')
    plt.ylabel('Énergie produite (kWh)')
    plt.title('Production d\'électricité - Historique et Prévisions')
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y %H:%M'))
    plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    # Enregistrement du graphique
    plt.savefig(file_path)  # Remplace automatiquement l'image existante
    plt.close()  # Ferme la figure pour libérer de la mémoire
    #plt.show()

def get_current_datetime():
    # Obtenez la date et l'heure actuelle
    current_datetime = datetime.now()

    # Troncature de l'heure à l'heure entière (ignorer les minutes et les secondes)
    current_datetime = current_datetime.replace(minute=0, second=0, microsecond=0)
    print("current_datetime:", current_datetime)
    return current_datetime # Retourne la date et l'heure actuelles

# Fonction pour préparer les données pour Prophet
def preparer_donnees_prophet(donnees_historiques):
    timestamps = donnees_historiques[0]['hourly']['time']
    temperatures = donnees_historiques[0]['hourly']['temperature_2m']
    df = pd.DataFrame({'ds': pd.to_datetime(timestamps), 'y': temperatures})
    return df

# Fonction pour entraîner le modèle Prophet et faire des prédictions
def entrainer_et_predire_prophet(df, future_steps=3*24):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=future_steps, freq='h')
    forecast = model.predict(future)
    return forecast

# Fonction pour afficher les résultats
def afficher_resultats_prophet(df, forecast):
    plt.figure(figsize=(12, 7))
    
    # Données et prédictions
    plt.plot(df['ds'], df['y'], label='Données observées')
    plt.plot(forecast['ds'], forecast['yhat'], label='Prédictions', color='orange')
    
    # Configuration des grilles
    plt.grid(True, which='both', axis='both')
    
    # Configuration de l'axe x
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    plt.gca().xaxis.set_minor_locator(mdates.DayLocator())
    
    # Configuration de l'axe y
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(5))
    
    # Labels et titre
    plt.xlabel('Temps')
    plt.ylabel('Température (°C)')
    plt.title('Prédictions de température pour les trois prochaines années')
    plt.legend()
    
    # Rotation des labels de l'axe x
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    #plt.show()

def predire_production_electricite_heure_par_heure(future_temperatures_list, prediction_dates_sunrise, prediction_times_sunrise, prediction_dates_sunset, prediction_times_sunset):
    predictions_future = []

    for i in range(len(future_temperatures_list)):
        timestamp, temperature = future_temperatures_list[i]
        # Convertir le Timestamp en string formaté
        date_str = timestamp.strftime('%Y-%m-%d')
        time_str = timestamp.strftime('%H:%M:%S')
        
        date = datetime.strptime(date_str, '%Y-%m-%d')
        time = datetime.strptime(time_str, '%H:%M:%S').time()

        # Déterminer l'ensoleillement
        sunrise_time = datetime.strptime(prediction_times_sunrise[i // 24], '%H:%M').time()
        sunset_time = datetime.strptime(prediction_times_sunset[i // 24], '%H:%M').time()

        ensoleillement_futur = 1.0 if sunrise_time <= time <= sunset_time else 0.0

        energie_produite_future = predire_production_electricite(
            puissance_nominale_par_panneau, nombre_de_panneaux, surface_par_panneau,
            efficacite_panneaux, ensoleillement_futur, inclinaison_panneaux,
            orientation_panneaux, facteur_de_performance, temperature)

        predictions_future.append((f"{date_str} {time_str}", energie_produite_future))

    return predictions_future
 
# Programme principal
if __name__ == "__main__":
    subprocess.run(['python', 'historique_donnees.py'], check=True)
    current_datetime = get_current_datetime()
    donnees_historiques = charger_donnees_historiques()
    if donnees_historiques is None:
        print("Aucune donnée historique trouvée. Arrêt du programme.")
        exit(1)

    temperatures, cloud_covers, sunrise, sunset, time_temperatures, time_sunrise, time_sunset = donnee_entrainement(donnees_historiques)

   # Obtenir et afficher les prédictions
    prediction_results_sunrise = predict_and_display_sunrise(time_sunrise, sunrise, current_datetime)
    prediction_results_sunset = predict_and_display_sunset(time_sunset, sunset, current_datetime)
    
    # Obtenir les prédictions pour les trois prochains jours
    prediction_dates_sunrise, prediction_times_sunrise, mean_predictions_sunrise, std_predictions_sunrise = predict_next_three_days_sunrise(time_sunrise, sunrise, current_datetime)
     # Obtenir les prédictions pour les trois prochains jours
    prediction_dates_sunset, prediction_times_sunset, mean_predictions_sunset, std_predictions_sunset = predict_next_three_days_sunset(time_sunset, sunset, current_datetime)
    print("prediction_dates_sunset", prediction_dates_sunset, "prediction_times_sunset", prediction_times_sunset, "mean_predictions_sunset", mean_predictions_sunset, "std_predictions_sunset", std_predictions_sunset)
    print("prediction_dates_sunrise", prediction_dates_sunrise, "prediction_times_sunrise", prediction_times_sunrise, "mean_predictions_sunrise", mean_predictions_sunrise, "std_predictions_sunrise", std_predictions_sunrise)
    # Afficher les prédictions avec les intervalles de confiance
    print("\nPrédictions du lever du soleil pour les trois prochains jours:")
    print("=" * 70)
    for i, (date, time) in enumerate(zip(prediction_dates_sunrise, prediction_times_sunrise)):
        confidence_interval = f"±{std_predictions_sunrise[i].item()*60:.0f} minutes"
        print(f"Date: {date} - Lever du soleil prévu à: {time} ({confidence_interval})")

    #afficher prediction sunset
    print("\nPrédictions du coucher du soleil pour les trois prochains jours:")
    print("=" * 70)
    for pred in prediction_results_sunset:
        print(f"Date: {pred['date']} - Coucher du soleil prévu à: {pred['time']} (±{pred['confidence']} minutes)")
    
    # Calculer la durée entre le lever et le coucher du soleil
    print("\nDurée entre le lever et le coucher du soleil pour les trois prochains jours:")
    print("=" * 70)
    durations = []
    for sunrise_date, sunrise_time, sunset_pred in zip(prediction_dates_sunrise, prediction_times_sunrise, prediction_results_sunset):
        # Convertir les temps en objets datetime (sans les secondes dans le format)
        sunrise_datetime = datetime.strptime(f"{sunrise_date} {sunrise_time}", "%Y-%m-%d %H:%M")
        sunset_datetime = datetime.strptime(f"{sunset_pred['date']} {sunset_pred['time']}", "%Y-%m-%d %H:%M")
        
        # Calculer la durée
        duration = sunset_datetime - sunrise_datetime
        durations.append(int(duration.total_seconds()))
        
        # Afficher la durée
        print(f"Date: {sunrise_date} - Durée prévue: {duration}")
    print("durations", durations)
    #sleep(50000)

    df = preparer_donnees_prophet(donnees_historiques)
    forecast = entrainer_et_predire_prophet(df)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    print("forecast.columns", forecast.columns)
    print("forecast", forecast)
    afficher_resultats_prophet(df, forecast)
    sleep(50000)
    # Extraire les prédictions de température pour les trois prochains jours
    future_temperatures = forecast[['ds', 'yhat']].tail(3*24)
    future_temperatures_list = future_temperatures.values.tolist()
    print("future_temperatures_list", future_temperatures_list)

    # Obtenir les données météorologiques actuelles
    data_meteo_actuelles_hourly = obtenir_donnees_meteo_actuelles_hourly(current_datetime)   
    mettre_a_jour_historique_hourly(data_meteo_actuelles_hourly)
    if current_datetime.hour==23: #mettre a jour avec les donnée de la journée
        data_meteo_actuelles_daily = obtenir_donnees_meteo_actuelles_daily(current_datetime)
        mettre_a_jour_historique_daily(data_meteo_actuelles_daily)
    

    # Calculer l'ensoleillement et la température actuels
    ensoleillement_actuel, temperature_actuelle = calculer_ensoleillement_et_temperature(data_meteo_actuelles_hourly)

    # Prédire la production d'électricité en temps réel
    energie_produite = predire_production_electricite(
        puissance_nominale_par_panneau, nombre_de_panneaux, surface_par_panneau,
        efficacite_panneaux, ensoleillement_actuel, inclinaison_panneaux,
        orientation_panneaux, facteur_de_performance, temperature_actuelle)

    # Stocker les nouvelles valeurs dans un fichier JSON
    stocker_donnees_json(energie_produite, current_datetime)
    # Afficher les résultats
    print(f"Ensoleillement actuel: {ensoleillement_actuel * 100}%")
    print(f"Température actuelle: {temperature_actuelle}°C")
    print(f"Énergie produite actuelle: {energie_produite:.2f} kWh")
    print(f"Timestamp: {current_datetime}")
    
    # Prédire la production d'électricité heure par heure pour les trois prochains jours
    predictions_future = predire_production_electricite_heure_par_heure(future_temperatures_list, prediction_dates_sunrise, prediction_times_sunrise, prediction_dates_sunset, prediction_times_sunset)
    # Afficher le graphique de la production d'électricité
    afficher_graphique(predictions_future)
# Afficher la durée d'ensoleillement pour les trois prochains jours
#stocker les nouvelles valeurs dans un json hourly et daily
#afficher un graphe de ce qui a ete produit jusqu'a present
#jouer avec les graphiques pour afficher genre le surplus d'energie produite ou 
#ce qui au contrainre a du etre achetrer pour combler le manque, l'argent economisé, etc
#utiliser une Régression linéaire bayésienne pour prédire la production d'électricité avec Pyro
#pour prédire la production d'électricité future, on va devoir prédire les variable de notre code qui ici sont l'ensoleillement moyen et la température moyenne
#pour predire ces variables, on vas utiliser les donnée sur ensoleillement moyen et la température moyenne et la duree d'ensoleillemnt, qu'on aura stoker a chaque fois qu'on va les recuperer
#Faire un site web pour afficher les données et les predictions
#afficher sur le passee l'ecart entre les prediction de production d'energie et la production d'energie reel