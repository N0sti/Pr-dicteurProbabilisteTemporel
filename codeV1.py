import os
import subprocess
import requests
import datetime
from datetime import datetime
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
def obtenir_donnees_meteo_actuelles():
    # Obtenez la date et l'heure actuelle
    current_datetime = datetime.now()

    # Troncature de l'heure à l'heure entière (ignorer les minutes et les secondes)
    current_datetime = current_datetime.replace(minute=0, second=0, microsecond=0)

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

def mettre_a_jour_historique(data):
    print("testststts", data)
    fichier_historique = 'donnees_historiques.json'
    # Charger le fichier historique
    with open(fichier_historique, 'r') as file:
        historique = json.load(file)
    print("data", data)
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
    print("historique",  historique[0]['hourly']['time'])
    # Sauvegarder les modifications dans le fichier historique
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

# Fonction pour stocker les nouvelles valeurs dans un fichier JSON dans le fichier historique
def stocker_donnees_json(energie_produite, timestamp):
    donnees = {
        "energie_produite": energie_produite,
        "timestamp": timestamp
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

# Programme principal
if __name__ == "__main__":
    
    subprocess.run(['python', 'historique_donnees.py'], check=True)  # Exécution du script et attente de la fin

    # Obtenir les données météorologiques actuelles
    data_meteo_actuelles = obtenir_donnees_meteo_actuelles()
    print("Type de data_meteo_actuelles:", type(data_meteo_actuelles))
    print("Contenu de data_meteo_actuelles:", data_meteo_actuelles)


    mettre_a_jour_historique(data_meteo_actuelles)

    # Calculer l'ensoleillement et la température actuels
    ensoleillement_actuel, temperature_actuelle = calculer_ensoleillement_et_temperature(data_meteo_actuelles)

    # Prédire la production d'électricité en temps réel
    energie_produite = predire_production_electricite(
        puissance_nominale_par_panneau, nombre_de_panneaux, surface_par_panneau,
        efficacite_panneaux, ensoleillement_actuel, inclinaison_panneaux,
        orientation_panneaux, facteur_de_performance, temperature_actuelle)

    # Obtenir le timestamp actuel
    timestamp = datetime.now().strftime('%d/%m/%Y %H:%M:%S')

    # Stocker les nouvelles valeurs dans un fichier JSON
    stocker_donnees_json(energie_produite, timestamp)

    # Afficher les résultats
    print(f"Ensoleillement actuel: {ensoleillement_actuel * 100}%")
    print(f"Température actuelle: {temperature_actuelle}°C")
    print(f"Énergie produite actuelle: {energie_produite:.2f} kWh")
    print(f"Timestamp: {timestamp}")

    # Afficher le graphique de la production d'électricité
    afficher_graphique()

#stocker les nouvelles valeurs dans un json
#afficher un phraphe de ce qui a ete produit jusqu'a present
#jouer avec les graphiques pour afficher genre le surplus d'energie produite ou 
#ce qui au contrainre a du etre achetrer pour combler le manque, l'argent economisé, etc
#utiliser une Régression linéaire bayésienne pour prédire la production d'électricité avec Pyro
#pour prédire la production d'électricité future, on va devoir prédire les variable de notre code qui ici sont l'ensoleillement moyen et la température moyenne
#pour predire ces variables, on vas utiliser les donnée sur ensoleillement moyen et la température moyenne, qu'on aura stoker a chaque fois qu'on va les recuperer
#Faire un site web pour afficher les données et les predictions
#afficher sur le passee l'ecart entre les prediction de production d'energie et la production d'energie reel