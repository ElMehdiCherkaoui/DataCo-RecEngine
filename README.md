# DataCo-RecEngine — Prévision de la consommation électrique sur Azure

## Présentation du projet

Ce projet consiste à développer un pipeline end-to-end de Data Engineering et de Machine Learning sur Microsoft Azure afin de prédire la consommation électrique horaire.

Le projet utilise des données historiques de consommation électrique de la région PJM East et couvre l'ensemble du processus, depuis l'ingestion des données jusqu'à l'entraînement, l'évaluation et la création d'un pipeline d'inférence.

## Technologies utilisées

- **Azure Data Factory** : ingestion et orchestration des données.
- **Azure Data Lake Storage Gen2** : stockage des données brutes.
- **Azure Machine Learning** : préparation, entraînement et évaluation des modèles.
- **Azure ML Designer** : construction des pipelines de Machine Learning.
- **Azure Compute** : exécution des traitements.
- **Python / Pandas** : nettoyage et feature engineering.
- **Terraform** : configuration et provisionnement de l'infrastructure Azure.

## Étapes réalisées

1. Création de l'infrastructure Azure.
2. Ingestion des fichiers CSV avec Azure Data Factory.
3. Organisation du Data Lake avec la structure `/raw/YYYY/MM/DD/`.
4. Création du datastore et du dataset dans Azure Machine Learning.
5. Analyse et nettoyage des données.
6. Création de variables temporelles : Hour, DayOfWeek, Month, Quarter, Year et DayOfYear.
7. Préparation des données pour l'entraînement et les tests.
8. Entraînement d'un modèle Boosted Decision Tree Regression.
9. Évaluation des performances avec MAE, RMSE et R².
10. Enregistrement du modèle dans Azure ML.
11. Création et validation du pipeline d'inférence.
12. Documentation de la gouvernance des données : qualité, sécurité, intégrité et traçabilité.

## Résultats du modèle

| Métrique | Résultat |
|---|---|
| MAE | 3 086,47 MW |
| RMSE | 4 072,02 MW |
| R² | 0,6062 |

Ces résultats proviennent de l'évaluation du modèle Boosted Decision Tree Regression.

## Gouvernance des données

Le projet intègre des principes de gouvernance visant à assurer :

- La qualité des données.
- La sécurité des accès et du stockage.
- La conservation des données brutes.
- La traçabilité des traitements.
- La documentation des sources de données.
- La prise en compte des principes du RGPD.

## Documentation

Le rapport complet du projet, comprenant les étapes de réalisation, les captures d'écran Azure et la gouvernance des données, est disponible ici :

**[Consulter le rapport PDF](docs/Rapport_Gouvernance_Donnees.pdf)**

## Conclusion

Ce projet a permis de mettre en œuvre une architecture Data et Machine Learning sur Microsoft Azure, de construire un modèle de prévision de la consommation électrique et de préparer son utilisation dans un pipeline d'inférence.

L'ensemble du travail a été documenté afin de faciliter sa compréhension et sa traçabilité.
