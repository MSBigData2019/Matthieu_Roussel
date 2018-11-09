Note sur l'instance du projet :

Les chemins d'accès aux fichiers doivent être modifiés pour tourner sur votre machine :

- Preprocessor.scala : 

ligne 50 : chemin d'accès au fichier train_clean.csv
ligne 22 : répertoire de destination prepared_trainingset

- Trainer.scala :

ligne 52 : strPathData => répertoire "prepared_trainingset" contenant le parquet pré-processé
ligne : strModele => répertoire de sauvegarde du modèle