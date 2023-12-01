import pandas as pd

# Chargement des données à partir du fichier CSV
data = pd.read_csv('dateset_financial.csv')

# Suppression des doublons
data.drop_duplicates(inplace=True)

# Convertir les colonnes appropriées de type string en float
columns_to_convert = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
data[columns_to_convert] = data[columns_to_convert].apply(pd.to_numeric, errors='coerce')

# Élimination des anomalies (par exemple, des valeurs négatives dans les colonnes de solde)
data = data[(data['oldbalanceOrg'] >= 0) & (data['newbalanceOrig'] >= 0) & (data['oldbalanceDest'] >= 0) & (data['newbalanceDest'] >= 0)]

# Normalisation des données (Min-Max scaling pour les colonnes spécifiées)
# columns_to_normalize = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
# data[columns_to_normalize] = (data[columns_to_normalize] - data[columns_to_normalize].min()) / (data[columns_to_normalize].max() - data[columns_to_normalize].min())

# Affichage des premières lignes pour comprendre la structure des données
print("Aperçu des données après nettoyage :")
print(data.head())

# Vérification des informations sur les colonnes et les types de données
print("\nInformations sur les colonnes après nettoyage :")
print(data.info())

# Vérification des statistiques descriptives pour identifier les anomalies
print("\nStatistiques descriptives après nettoyage :")
print(data.describe())

# Sauvegarde des données nettoyées dans un nouveau fichier CSV
data.to_csv('dateset_financial_clean.csv', index=False)
