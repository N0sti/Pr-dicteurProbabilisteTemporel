import requests
import datetime
import json
import os

# Clé API OpenWeatherMap (remplacez par votre propre clé)
CITY = 'cachan'

def obtenir_donnees_historiques(CITY, start, end):
    # Convertir les dates en format attendu (par exemple, 'YYYY-MM-DD')
    start_date = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d')
    end_date = datetime.datetime.fromtimestamp(end).strftime('%Y-%m-%d')

    # Construire l'URL avec les dates dynamiques
    url = f'https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=48.7833&longitude=2.3333&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high&daily=sunrise,sunset,daylight_duration,sunshine_duration&timezone=Europe%2FBerlin'
    print(url)

    # Effectuer la requête
    response = requests.get(url)

    # Vérification du statut de la réponse (si la requête a réussi)
    if response.status_code != 200:
        print(f"Erreur lors de la récupération des données: {response.status_code}")
        print(f"Message d'erreur: {response.text}")
        return None

    return response.json()

def stocker_donnees_historiques(data, fichier):
    if data is None:
        print("Aucune donnée à stocker.")
        return

    # Vérifier si le fichier existe déjà et contient des données
    if os.path.exists(fichier):
        with open(fichier, 'r') as f:
            try:
                contenu = json.load(f)
                if contenu:  # Vérifier si le fichier n'est pas vide ou déjà rempli
                    print(f"Le fichier {fichier} contient déjà des données. Aucune donnée ne sera ajoutée.")
                    return
            except json.JSONDecodeError:
                print(f"Erreur lors de la lecture du fichier {fichier}. Il sera écrasé.")

    # Créer ou écraser le fichier et stocker les données
    with open(fichier, 'w') as f:
        json.dump([data], f, indent=4)
    print(f"Données stockées dans le fichier {fichier}.")

if __name__ == "__main__":
    end_time = int(datetime.datetime.now().timestamp())  # Timestamp actuel (fin de la période)
    start_time = end_time - 1 * 365 * 24 * 3600  # 1 an en secondes (début de la période)
    print("start_time:", start_time)
    print("end_time:", end_time)

    # Récupérer toutes les données pour les deux dernières années
    donnees_historiques = obtenir_donnees_historiques(CITY, start_time, end_time)

    # Stocker les données dans un fichier JSON
    stocker_donnees_historiques(donnees_historiques, 'donnees_historiques.json')
