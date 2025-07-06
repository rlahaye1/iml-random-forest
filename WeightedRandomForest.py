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
        self.classes_ = np.unique(y)

        for _ in range(n_trees):
            # Tirage aléatoire avec remplacement
            X_sample, y_sample = resample(X, y)
            tree = Tree()
            self.add_tree(tree, X_sample, y_sample, weight=1.0)

    def add_tree(self, tree, X=None, y=None, weight=1.0):
        if X is not None and y is not None:
            tree.fit(X, y)
            tree.classes_ = self.classes_

        self.trees.append(tree)
        self.weights.append(weight)


    # def predict_proba(self, X):
    #     if not self.trees:
    #         raise ValueError("Aucun arbre n'a été ajouté à la forêt.")
    #     total_weight = sum(self.weights)
    #     probas = np.zeros((X.shape[0], len(self.classes_)))

    #     for tree, weight in zip(self.trees, self.weights):
    #         probas += weight * tree.predict_proba(X)

    #     return probas / total_weight
    
    def predict_proba(self, X):
        if not self.trees:
            raise ValueError("Aucun arbre n'a été ajouté à la forêt.")

        total_weight = sum(self.weights)
        probas = np.zeros((X.shape[0], len(self.classes_)))

        for tree, weight in zip(self.trees, self.weights):
            local_proba = tree.predict_proba(X)
            aligned_proba = np.zeros((X.shape[0], len(self.classes_)))

            for i, cls in enumerate(tree.classes_):
                col_index = np.where(self.classes_ == cls)[0][0]
                aligned_proba[:, col_index] = local_proba[:, i]

            probas += weight * aligned_proba

        return probas / total_weight

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
