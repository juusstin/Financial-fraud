# Import des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

# Charger le dataset
dataset_path = "dateset_financial.csv"
df = pd.read_csv(dataset_path)

# Sélectionner un sous-ensemble de données (par exemple, les 1000 premières lignes)
subset_size = 30000
df_subset = df.head(subset_size)

# Séparer les caractéristiques (X) de la cible (y)
X = df_subset.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = df_subset['isFraud']

# Utiliser ColumnTransformer pour appliquer l'encodage one-hot aux colonnes catégorielles
categorical_cols = ['type', 'nameOrig', 'nameDest']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)
    ],
    remainder='passthrough'
)

# Appliquer l'encodage one-hot aux données
X_encoded = preprocessor.fit_transform(X)

# Réduire la dimensionnalité avec PCA (exemple avec 50 composantes)
pca = PCA(n_components=50)
X_encoded_pca = pca.fit_transform(X_encoded.toarray())

# Diviser le dataset en ensemble d'entraînement et ensemble de test
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_encoded_pca, y, test_size=0.2, random_state=2)

# Créer un modèle de forêt aléatoire avec moins d'arbres et parallélisation
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Entraîner le modèle sur l'ensemble d'entraînement encodé et réduit en dimensionnalité
random_forest_model.fit(X_train_pca, y_train)

# Prédire les labels sur l'ensemble de test encodé et réduit en dimensionnalité
y_pred = random_forest_model.predict(X_test_pca)

# Évaluer les performances du modèle
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

# Afficher les résultats
print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:\n', classification_report_result)
