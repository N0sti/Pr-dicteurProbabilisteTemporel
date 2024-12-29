# Projet de Prévision de Production d'Électricité Solaire

## But du Projet

Ce projet a pour objectif de prédire et de suivre la production d'électricité d'un système photovoltaïque en utilisant des données météorologiques en temps réel. Il permet également de stocker ces informations dans un fichier historique, de les analyser au fil du temps et d'afficher des graphiques pour visualiser les tendances de la production d'énergie solaire.

Le système utilise une combinaison de données ensoleillement, température et autres variables météorologiques pour prédire la quantité d'énergie produite. En utilisant ces prédictions, il peut aussi estimer le surplus d'énergie produit, l'énergie manquante, et calculer des économies réalisées en fonction de la production d'énergie par rapport à une consommation estimée.

### Fonctionnalités principales

- **Prédiction de la production d'énergie** : Utilisation des données météorologiques (ensoleillement, température) pour prédire la production d'électricité d'un système photovoltaïque.
- **Stockage des données** : Enregistrement des données de production d'énergie, de l'ensoleillement et de la température dans un fichier JSON historique.
- **Affichage des résultats** : Génération de graphiques montrant la production d'énergie au fil du temps pour une analyse visuelle.
- **Prédictions futures** : Modélisation des prévisions de production d'énergie à l'aide de méthodes de régression linéaire bayésienne, en utilisant des données historiques pour estimer les tendances futures.
- **Site Web pour Visualisation** : Une interface web pour consulter les données de production, les prévisions futures, et les comparaisons avec la production réelle.
- **Analyse des économies d'énergie** : Estimation de l'énergie économisée ou de l'énergie qui aurait dû être achetée pour combler un manque de production.

### Fonctionnement du projet

1. **Données météorologiques** : Le projet récupère les données météorologiques actuelles (température et couverture nuageuse) à l'aide d'une API tierce (Open-Meteo).
   
2. **Calcul de la production d'énergie** : Sur la base de la puissance nominale des panneaux photovoltaïques, de leur efficacité, et des données météorologiques, le projet calcule la production d'énergie horaire du système.

3. **Stockage des données** : Les résultats sont enregistrés dans un fichier JSON qui conserve l'historique de la production d'énergie, de l'ensoleillement et de la température.

4. **Affichage graphique** : À partir des données historiques, un graphique est généré pour visualiser la production d'énergie au fil du temps.

5. **Prédiction avec Régression Linéaire Bayésienne** : Le projet utilise la régression linéaire bayésienne pour prédire la production d'énergie future, en se basant sur les tendances observées des variables d'ensoleillement et de température.

6. **Interface Web** : Un site web est prévu pour afficher en temps réel les données météorologiques, la production d'énergie actuelle et les prédictions pour les heures à venir.

### Technologies utilisées

- **Python** : Le langage principal utilisé pour récupérer les données, effectuer les calculs, et gérer les fichiers JSON.
- **Matplotlib** : Utilisé pour afficher des graphiques de la production d'énergie.
- **Pyro** : Utilisé pour effectuer les prédictions avec régression linéaire bayésienne.
- **Flask/Django (prévu)** : Un framework web pour créer l'interface utilisateur permettant de visualiser les données en temps réel.
- **API Open-Meteo** : Pour récupérer les données météorologiques en temps réel.

### Instructions d'installation

1. Clonez le repository :
   ```bash
   git clone https://github.com/N0sti/Pr-dicteurProbabilisteTemporel.git
   cd projet-production-solaire
