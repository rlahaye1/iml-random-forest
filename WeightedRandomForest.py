import numpy as np
from Tree import Tree
from sklearn.utils import resample

class WeightedRandomForest:
    def __init__(self):
        self.trees = []
        self.weights = []
        self.classes_ = None  # pour garder une référence commune

    def fit(self, X, y, n_trees=10):
        """
        Initialise n arbres entraînés chacun sur un échantillon bootstrapé (tiré aléatoirement)
        de X, y avec la classe Tree.
        """
        for _ in range(n_trees):
            # Tirage aléatoire avec remplacement
            X_sample, y_sample = resample(X, y)
            tree = Tree()
            self.add_tree(tree, X_sample, y_sample, weight=1.0)

    def add_tree(self, tree, X=None, y=None, weight=1.0):
        # Si les données sont fournies, on entraîne l'arbre
        if X is not None and y is not None:
            tree.fit(X, y)

        # On s'assure que l'arbre peut prédire les probabilités
        if not hasattr(tree, "predict_proba"):
            raise ValueError("Le tree doit avoir une méthode predict_proba")

        # On stocke les classes lors du premier ajout
        if self.classes_ is None:
            self.classes_ = tree.classes_
        else:
            # Vérifie que les classes sont les mêmes
            if not np.array_equal(self.classes_, tree.classes_):
                raise ValueError("Les classes de l'arbre ne correspondent pas à celles déjà présentes.")

        self.trees.append(tree)
        self.weights.append(weight)

    def predict_proba(self, X):
        if not self.trees:
            raise ValueError("Aucun arbre n'a été ajouté à la forêt.")
        total_weight = sum(self.weights)
        probas = np.zeros((X.shape[0], len(self.classes_)))

        for tree, weight in zip(self.trees, self.weights):
            probas += weight * tree.predict_proba(X)

        return probas / total_weight

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
