import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Charger le dataset
data = pd.read_csv("dateset_financial_clean.csv")

# Supprimer les colonnes non nécessaires
data = data.drop(['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)

# Convertir les variables catégorielles en variables indicatrices
data = pd.get_dummies(data, columns=['type'], drop_first=True)

# Séparer les caractéristiques (X) de la cible (y)
X = data.drop('isFraud', axis=1)
y = data['isFraud']

# Diviser le dataset en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardiser les caractéristiques
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Créer et entraîner le modèle de régression logistique avec un nombre d'itérations maximal plus élevé
model = LogisticRegression(max_iter=1000)  # Augmenter le nombre d'itérations
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer les performances du modèle
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
print("\nRapport de classification :\n", classification_report(y_test, y_pred))
