# Code permettant d'analyser les préliminaires de données

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le fichier CSV nettoyé
df = pd.read_csv('dateset_financial_clean.csv')

#---------- DEBUT ANALYSE PRELIMINAIRE DES DONNEES ------------#
# Afficher les informations générales sur le DataFrame
print(df.info())

# Afficher un aperçu des premières lignes du DataFrame
print(df.head())

# Résumé statistique des colonnes numériques
print(df.describe())

# Visualisation des corrélations entre les colonnes numériques
correlation_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Matrice de corrélation')
plt.show()

# Visualisation de la distribution des montants
plt.figure(figsize=(10, 6))
sns.histplot(df['amount'], bins=50, kde=True, color='blue')
plt.title('Distribution des montants')
plt.xlabel('Montant')
plt.ylabel('Fréquence')
plt.show()

# Visualisation de la relation entre le montant et la variable cible (isFraud)
plt.figure(figsize=(10, 6))
sns.boxplot(x='isFraud', y='amount', data=df)
plt.title('Relation entre le montant et la fraude')
plt.xlabel('Fraude')
plt.ylabel('Montant')
plt.show()
#---------- FIN ANALYSE PRELIMINAIRE DES DONNEES ------------#
