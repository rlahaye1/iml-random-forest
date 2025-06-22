import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from WeightedRandomForest import WeightedRandomForest
from Tree import Tree  # ta classe custom

def main():
    # Charger le dataset
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)

    # Créer la forêt avec 3 arbres initiaux
    forest = WeightedRandomForest()
    forest.fit(X_train, y_train, n_trees=3)

    # Prédiction initiale
    print("Prédictions initiales :")
    print(forest.predict(X_test[:5]))

    # Ajouter un arbre très spécifique avec un poids élevé
    X_subset = X_train[:10]
    y_subset = y_train[:10]
    important_tree = Tree()
    forest.add_tree(important_tree, X_subset, y_subset, weight=5.0)

    # Nouvelle prédiction
    print("Prédictions après ajout de l'arbre pondéré :")
    print(forest.predict(X_test[:5]))

if __name__ == "__main__":
    main()
