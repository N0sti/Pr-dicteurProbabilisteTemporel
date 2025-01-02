# Projet de Prévision de Production d'Électricité Solaire

## Objectif du Projet

Le projet vise à prédire la production d'électricité d'un système photovoltaïque en utilisant des données météorologiques actuelles et historiques. Il inclut des fonctionnalités permettant de récupérer des données météorologiques, de mettre à jour un historique de données, et de prédire la production d'énergie future à l'aide de modèles de prédiction. Le système analyse également la production d'électricité pour estimer le surplus ou le manque d'énergie, et fournit une visualisation des résultats à travers une interface web. Le tout est conçu pour tourner sur un **Raspberry Pi 5** servant de serveur pour l'exécution et l'affichage des résultats.

### Fonctionnalités principales

- **Prédiction de la production d'énergie** : Utilisation des données météorologiques (ensoleillement, température, couverture nuageuse) pour prédire la production d'électricité des panneaux photovoltaïques.
- **Récupération des données météorologiques** : Intégration de l'API Open-Meteo pour récupérer les données météorologiques en temps réel (température, couverture nuageuse, etc.).
- **Mise à jour de l'historique** : Enregistrement des données météorologiques et de la production d'énergie dans un fichier JSON pour un suivi historique.
- **Modèles de prédiction de la production** : Utilisation de modèles comme la régression linéaire bayésienne et des modèles sinusoïdaux pour estimer la production d'électricité future.
- **Prédiction des heures de lever et coucher du soleil** : Modélisation des heures de lever et coucher du soleil à l'aide de fonctions sinusoïdales, prédisant les événements pour les trois prochains jours.
- **Interface Web** : Une interface web développée avec Flask pour afficher en temps réel les graphiques de production d'énergie et les prédictions.
- **Analyse des économies d'énergie** : Estimation de l'énergie économisée par rapport à la consommation estimée et comparaison avec une production d'énergie de référence.
- **Rafraîchissement horaire des graphiques** : Les graphiques de production d'énergie sont automatiquement mis à jour toutes les heures pour afficher les dernières prévisions et la production réelle en temps réel.

### Fonctionnement du projet

1. **Raspberry Pi 5 comme serveur** : Le projet est conçu pour être exécuté sur un Raspberry Pi 5 qui agit comme serveur principal. Ce serveur gère la récupération des données, les calculs, la mise à jour des graphiques et l'interface web.
   
2. **Récupération des données météorologiques** : Le projet récupère les données météorologiques actuelles à partir de l'API Open-Meteo pour obtenir la température, la couverture nuageuse, etc.

3. **Mise à jour de l'historique** : Les nouvelles données météorologiques sont ajoutées à un fichier JSON pour un suivi détaillé des conditions météo et de la production d'énergie.

4. **Calcul de la production d'énergie** : Sur la base des données météorologiques actuelles et de la puissance nominale des panneaux solaires, le système prédit la production d'énergie horaire.

5. **Prédiction des heures de lever et coucher du soleil** : À l'aide de modèles sinusoïdaux (pour le lever et le coucher du soleil), le projet prédit les horaires des deux événements pour les trois prochains jours.

6. **Prédiction de la production future** : Utilisation de la régression linéaire bayésienne et de modèles basés sur les données historiques et météorologiques pour estimer la production future d'électricité.

7. **Affichage des résultats** : Les résultats de la prédiction et de la production d'énergie sont affichés sous forme de graphiques interactifs via une interface web.

8. **Rafraîchissement horaire des graphiques** : Le système est conçu pour mettre à jour automatiquement les graphiques toutes les heures, en récupérant de nouvelles données et en affichant des prévisions et résultats actualisés.

9. **Exécution périodique** : Le système fonctionne de manière périodique sur le Raspberry Pi, garantissant des prévisions à jour et un suivi continu de la production d'énergie.

### Technologies utilisées

- **Python** : Le langage principal utilisé pour récupérer les données, effectuer les calculs et gérer les fichiers JSON.
- **Matplotlib** : Utilisé pour l'affichage des graphiques de production d'énergie.
- **Pyro** : Utilisé pour effectuer les prédictions avec régression linéaire bayésienne.
- **Prophet** : Utilisé pour les prédictions de température, un modèle de prévision de séries temporelles.
- **Flask** : Framework web pour créer l'interface utilisateur permettant de visualiser les données en temps réel.
- **API Open-Meteo** : Pour récupérer les données météorologiques en temps réel.
- **Modèle sinusoïdal** : Utilisé pour prédire les heures de lever et coucher du soleil.
- **Raspberry Pi 5** : Le serveur exécutant le projet, gérant la récupération des données, les calculs et la mise à jour des graphiques.

### Structure des Fonctions

- **Récupération des données météorologiques**
  - **obtenir_donnees_meteo_actuelles_hourly** : Récupère les données météorologiques actuelles à l'échelle horaire.
  - **obtenir_donnees_meteo_actuelles_daily** : Récupère les données météorologiques actuelles à l'échelle journalière.

- **Mise à jour de l'historique des données**
  - **mettre_a_jour_historique_hourly** : Met à jour l'historique des données météorologiques à l'échelle horaire dans un fichier JSON.
  - **mettre_a_jour_historique_daily** : Met à jour l'historique des données météorologiques à l'échelle journalière.

- **Calcul de l'ensoleillement et de la température actuels**
  - **calculer_ensoleillement_et_temperature** : Calcule l'ensoleillement actuel en fonction de la couverture nuageuse et extrait la température actuelle.

- **Prédiction de la production d'énergie**
  - **predire_production_electricite** : Prédit la production d'électricité en fonction des paramètres des panneaux solaires et des conditions météorologiques.

- **Prédiction des heures de lever et coucher du soleil**
  - **model_sunrise_sinusoidal** : Modèle sinusoïdal pour prédire l'heure du lever du soleil.
  - **model_sunset_sinusoidal** : Modèle sinusoïdal pour prédire l'heure du coucher du soleil.
  - **predict_next_three_days_sunrise** : Prédit les heures de lever du soleil pour les trois prochains jours.
  - **predict_next_three_days_sunset** : Prédit les heures de coucher du soleil pour les trois prochains jours.

- **Affichage des résultats**
  - **predict_and_display_sunrise** : Affiche les prédictions du lever du soleil.
  - **predict_and_display_sunset** : Affiche les prédictions du coucher du soleil.
  - **afficher_graphique** : Affiche un graphique de la production d'énergie.

- **Stockage des données historiques**
  - **stocker_donnees_json** : Enregistre les nouvelles données de production d'énergie dans un fichier JSON.

- **Interface Web**
  - **app.py** : Fichier Flask pour l'interface web, permettant d'afficher les graphiques de production d'énergie et les prédictions.

### Instructions d'installation

1. Clonez le repository :
   ```bash
   git clone https://github.com/N0sti/Pr-dicteurProbabilisteTemporel.git
   cd projet-production-solaire
2. Import nécessaire
   Python est un prérequis pour faire tourner le code
   ```bash
   pip install requests matplotlib numpy torch pandas prophet pyro-ppl
   ```
3. Faire tourner le code
   ```bash
   python app.py
