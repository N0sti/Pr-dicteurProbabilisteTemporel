import os
import subprocess
import requests
import datetime
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt



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
    print("data", data)
    
    return data

# Fonction pour calculer l'ensoleillement et la température actuels
def calculer_ensoleillement_et_temperature(data):
    print(data)  # Débogage pour afficher les données reçues
    
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
    print("testststts", data)
    fichier_historique = 'donnees_historiques.json'
    # Charger le fichier historique
    with open(fichier_historique, 'r') as file:
        historique = json.load(file)
    #print("data", data)
    print(data['hourly']['time'])
    print(data['hourly']['temperature_2m'])
    print(data['hourly']['cloud_cover'])
    print(data['hourly']['cloud_cover_low'])
    print(data['hourly']['cloud_cover_mid'])
    print(data['hourly']['cloud_cover_high'])
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
    print("testststts", data)
    fichier_historique = 'donnees_historiques.json'
    # Charger le fichier historique
    with open(fichier_historique, 'r') as file:
        historique = json.load(file)
    #print("data", data)
    print(data['daily']['time'])
    print(data['daily']['sunrise'])
    print(data['daily']['sunset'])
    print(data['daily']['daylight_duration'])
    print(data['daily']['sunshine_duration'])
    # Ajouter les nouvelles données à la section "hourly" de l'historique
    #print("historique",  historique[0]['daily']['time'])
    for item in data['daily']['time']:
        historique[0]['daily']['time'].append(item)
    #print("historique",  historique[0]['daily']['time'])

    for item in data['daily']['sunrise']:
        historique[0]['daily']['sunrise'].append(item)
    print("historique",  historique[0]['daily']['sunrise'])

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

def filtrer_donnees_mois_precedent(donnees_historiques):
    timestamps = donnees_historiques[0]['hourly']['time']
    temperatures = donnees_historiques[0]['hourly']['temperature_2m']
    cloud_covers = donnees_historiques[0]['hourly']['cloud_cover']

    # Calculer la date du premier jour du mois précédent
    today = datetime.now()
    first_day_of_previous_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)

    filtered_timestamps = []
    filtered_temperatures = []
    filtered_cloud_covers = []

    for i in range(len(timestamps)):
        timestamp = datetime.strptime(timestamps[i], '%Y-%m-%dT%H:%M')
        if timestamp >= first_day_of_previous_month:
            filtered_timestamps.append(timestamps[i])
            filtered_temperatures.append(temperatures[i])
            filtered_cloud_covers.append(cloud_covers[i])

    return filtered_timestamps, filtered_temperatures, filtered_cloud_covers

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

    print(f"Nouvelle donnée ajoutée: {donnees}")

def charger_donnees_historiques():
    try:
        with open('donnees_historiques.json', 'r') as fichier:
            donnees = json.load(fichier)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Aucune donnée valide trouvée pour afficher le graphique.")
        return None
    return donnees

def calculer_production_quotidienne(filtered_timestamps, filtered_temperatures, filtered_cloud_covers):
    production_quotidienne = {}
    current_date = None
    daily_production = 0

    for i in range(len(filtered_timestamps)):
        timestamp = filtered_timestamps[i]
        date = timestamp.split('T')[0]
        temperature_actuelle = filtered_temperatures[i]
        cloud_cover = filtered_cloud_covers[i]
        ensoleillement_actuel = (100 - cloud_cover) / 100
        energie_produite = predire_production_electricite(
            puissance_nominale_par_panneau, nombre_de_panneaux, surface_par_panneau,
            efficacite_panneaux, ensoleillement_actuel, inclinaison_panneaux,
            orientation_panneaux, facteur_de_performance, temperature_actuelle)

        if current_date is None:
            current_date = date

        if date == current_date:
            daily_production += energie_produite
        else:
            production_quotidienne[current_date] = daily_production
            current_date = date
            daily_production = energie_produite

    if current_date:
        production_quotidienne[current_date] = daily_production

    return production_quotidienne

def afficher_graphique_quotidien(production_quotidienne):
    dates = list(production_quotidienne.keys())
    production = list(production_quotidienne.values())

    # Formater les dates au format 30/09/2024
    formatted_dates = [datetime.strptime(date, '%Y-%m-%d').strftime('%d/%m/%Y') for date in dates]
    print("formatted_dates", formatted_dates)

    plt.figure(figsize=(10, 6))
    plt.plot(formatted_dates, production, marker='o', linestyle='-', color='blue', label='Énergie produite')
    plt.xlabel('Date')
    plt.ylabel('Énergie produite (kWh)')
    plt.title('Production d\'électricité quotidienne')
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    timestamps = donnees_historiques[0]['hourly']['time']
    temperatures = donnees_historiques[0]['hourly']['temperature_2m']
    cloud_covers = donnees_historiques[0]['hourly']['cloud_cover']

    production_mensuelle = []

    for i in range(len(timestamps)):
        temperature_actuelle = temperatures[i]
        cloud_cover = cloud_covers[i]
        ensoleillement_actuel = (100 - cloud_cover) / 100
        energie_produite = predire_production_electricite(
            puissance_nominale_par_panneau, nombre_de_panneaux, surface_par_panneau,
            efficacite_panneaux, ensoleillement_actuel, inclinaison_panneaux,
            orientation_panneaux, facteur_de_performance, temperature_actuelle)
        production_mensuelle.append(energie_produite)
    print("production_mensuelle", production_mensuelle)

    return timestamps, production_mensuelle
# Fonction pour afficher un graphique de la production d'électricité
def afficher_graphique():
    try:
        with open('donnees_graphs.json', 'r') as fichier:
            donnees = json.load(fichier)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Aucune donnée valide trouvée pour afficher le graphique.")
        return

    if not donnees:
        print("Le fichier JSON ne contient aucune donnée.")
        return

    timestamps = []
    energie_produite = []

    for entry in donnees:
        if 'timestamp' in entry and 'energie_produite' in entry:
            timestamps.append(entry['timestamp'])
            energie_produite.append(entry['energie_produite'])

    if not timestamps or not energie_produite:
        print("Aucune donnée valide à afficher.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, energie_produite, marker='o', linestyle='-', color='blue', label='Énergie produite')
    plt.xlabel('Temps')
    plt.ylabel('Énergie produite (kWh)')
    plt.title('Production d\'électricité au fil du temps')
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    try:
        with open('donnees_historiques.json', 'r') as fichier:
            donnees = json.load(fichier)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Aucune donnée valide trouvée pour afficher le graphique.")
        return

    if not donnees:
        print("Le fichier JSON ne contient aucune donnée.")
        return

    # Liste des timestamps et énergies produites
    timestamps = []
    energie_produite = []

    # Ajouter une vérification pour garantir que 'timestamp' et 'energie_produite' existent dans chaque entrée
    for entry in donnees:
        if 'timestamp' in entry and 'energie_produite' in entry:
            timestamps.append(entry['timestamp'])
            energie_produite.append(entry['energie_produite'])

    if not timestamps or not energie_produite:
        print("Aucune donnée valide à afficher.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, energie_produite, marker='o', linestyle='-', color='blue', label='Énergie produite')
    plt.xlabel('Temps')
    plt.ylabel('Énergie produite (kWh)')
    plt.title('Production d\'électricité au fil du temps')
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_current_datetime():
    # Obtenez la date et l'heure actuelle
    current_datetime = datetime.now()

    # Troncature de l'heure à l'heure entière (ignorer les minutes et les secondes)
    current_datetime = current_datetime.replace(minute=0, second=0, microsecond=0)
    print("current_datetime:", current_datetime)
    return current_datetime # Retourne la date et l'heure actuelles

# Programme principal
if __name__ == "__main__":
    
    subprocess.run(['python', 'historique_donnees.py'], check=True)  # Exécution du script et attente de la fin
    current_datetime=get_current_datetime()
    # Obtenir les données météorologiques actuelles
    data_meteo_actuelles_hourly = obtenir_donnees_meteo_actuelles_hourly(current_datetime)
    print("Type de data_meteo_actuelles:", type(data_meteo_actuelles_hourly))
    print("Contenu de data_meteo_actuelles:", data_meteo_actuelles_hourly)
    mettre_a_jour_historique_hourly(data_meteo_actuelles_hourly)
    if current_datetime.hour==23: #mettre a jour avec les donnée de la journée
        data_meteo_actuelles_daily = obtenir_donnees_meteo_actuelles_daily(current_datetime)
        print("Type de data_meteo_actuelles:", type(data_meteo_actuelles_daily))
        print("Contenu de data_meteo_actuelles:", data_meteo_actuelles_daily)
        mettre_a_jour_historique_daily(data_meteo_actuelles_daily)
  

    # Calculer l'ensoleillement et la température actuels
    ensoleillement_actuel, temperature_actuelle = calculer_ensoleillement_et_temperature(data_meteo_actuelles_hourly)

    # Prédire la production d'électricité en temps réel
    energie_produite = predire_production_electricite(
        puissance_nominale_par_panneau, nombre_de_panneaux, surface_par_panneau,
        efficacite_panneaux, ensoleillement_actuel, inclinaison_panneaux,
        orientation_panneaux, facteur_de_performance, temperature_actuelle)

    # Obtenir le timestamp actuel
    #timestamp = datetime.now().strftime('%d/%m/%Y %H:%M:%S')

    # Stocker les nouvelles valeurs dans un fichier JSON
    #stocker_donnees_json(energie_produite, timestamp)
    stocker_donnees_json(energie_produite, current_datetime)
    # Afficher les résultats
    print(f"Ensoleillement actuel: {ensoleillement_actuel * 100}%")
    print(f"Température actuelle: {temperature_actuelle}°C")
    print(f"Énergie produite actuelle: {energie_produite:.2f} kWh")
    print(f"Timestamp: {current_datetime}")
    # Afficher le graphique de la production d'électricité
    afficher_graphique()
    
    donnees_historiques = charger_donnees_historiques()
    if donnees_historiques:
        filtered_timestamps, filtered_temperatures, filtered_cloud_covers = filtrer_donnees_mois_precedent(donnees_historiques)
        production_quotidienne = calculer_production_quotidienne(filtered_timestamps, filtered_temperatures, filtered_cloud_covers)
        afficher_graphique_quotidien(production_quotidienne)

#stocker les nouvelles valeurs dans un json hourly et daily
#afficher un graphe de ce qui a ete produit jusqu'a present
#jouer avec les graphiques pour afficher genre le surplus d'energie produite ou 
#ce qui au contrainre a du etre achetrer pour combler le manque, l'argent economisé, etc
#utiliser une Régression linéaire bayésienne pour prédire la production d'électricité avec Pyro
#pour prédire la production d'électricité future, on va devoir prédire les variable de notre code qui ici sont l'ensoleillement moyen et la température moyenne
#pour predire ces variables, on vas utiliser les donnée sur ensoleillement moyen et la température moyenne et la duree d'ensoleillemnt, qu'on aura stoker a chaque fois qu'on va les recuperer
#Faire un site web pour afficher les données et les predictions
#afficher sur le passee l'ecart entre les prediction de production d'energie et la production d'energie reel