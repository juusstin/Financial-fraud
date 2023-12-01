import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Fonction d'activation (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Dérivée de la fonction d'activation sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Fonction pour convertir les probabilités en classes (fraude ou non-fraude)
def convert_to_class(probabilities, threshold=0.5):
    return (probabilities > threshold).astype(int)

# Fonction d'entraînement du réseau de neurones
def train_network(inputs, outputs, epochs):
    # Initialisation des poids
    np.random.seed(1)
    input_layer_size = len(inputs[0])
    output_layer_size = len(outputs[0])
    hidden_layer_size = 10

    weights_input_hidden = 2 * np.random.random((input_layer_size, hidden_layer_size)) - 1
    weights_hidden_output = 2 * np.random.random((hidden_layer_size, output_layer_size)) - 1

    # Entraînement du réseau
    for epoch in range(epochs):
        # Forward pass
        hidden_layer_input = np.dot(inputs, weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
        predicted_output = sigmoid(output_layer_input)

        # Calcul de l'erreur
        error = outputs - predicted_output

        # Backpropagation
        output_error = error * sigmoid_derivative(predicted_output)
        hidden_layer_error = output_error.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)

        # Mise à jour des poids
        weights_hidden_output += hidden_layer_output.T.dot(output_error)
        weights_input_hidden += inputs.T.dot(hidden_layer_error)

    return predicted_output

# Charger le dataset
df = pd.read_csv("dataset_financial_clean.csv")

# Extraire les colonnes nécessaires
inputs = df[["oldbalanceOrg"]].values
outputs = df[["isFraud"]].values

# Normaliser les données
scaler = MinMaxScaler()
inputs = scaler.fit_transform(inputs)

# Entraînement du réseau sur 10000 itérations
epochs = 100
predicted_output = train_network(inputs, outputs, epochs)

# Convertir les probabilités en classes (fraude ou non-fraude)
predicted_classes = convert_to_class(predicted_output)

# Créer un DataFrame avec les résultats
results_df = pd.DataFrame({
    "Données d'entrée normalisées": inputs.flatten(),
    "Prédictions du réseau (probabilités de fraude)": predicted_output.flatten(),
    "Classes prédites (Fraude/Non-fraude)": predicted_classes.flatten(),
    "Étiquettes réelles (isFraud)": outputs.flatten(),
})

# Ajuster les paramètres d'affichage pour limiter le nombre de lignes
pd.set_option("display.max_rows", 10)  # Définir le nombre maximum de lignes à afficher
pd.set_option("display.max_columns", None)  # Afficher toutes les colonnes

# Afficher le DataFrame sous forme de tableau
print(results_df)
