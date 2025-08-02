import numpy as np
from Tree import Tree
from sklearn.utils import resample

class WeightedRandomForest:
    def __init__(self):
        """
        Initialise une forêt aléatoire pondérée vide.
        - trees : liste des arbres de décision.
        - weights : poids associés à chaque arbre.
        - classes_ : liste des classes rencontrées à l'entraînement.
        - per_tree_preds : prédictions par arbre (utilisées pour analyse).
        - accuracy : liste pour stocker la précision par arbre si nécessaire.
        """
        self.trees = []
        self.weights = []
        self.classes_ = None
        self.per_tree_preds = []
        self.accuracy= []

    def fit(self, X, y, n_trees=10):
        """
        Entraîne la forêt avec n arbres, chacun sur un échantillon bootstrapé de X et y.

        Paramètres :
        - X : array-like, features d'entrée.
        - y : array-like, étiquettes cibles.
        - n_trees : int, nombre d'arbres à ajouter à la forêt (par défaut : 10).
        """
        self.classes_ = np.unique(y)

        for _ in range(n_trees):
            # Tirage aléatoire avec remplacement
            X_sample, y_sample = resample(X, y)
            tree = Tree()
            self.add_tree(tree, X_sample, y_sample, weight=0.1)

    def add_tree(self, tree, X=None, y=None, weight=0.1, forced_features=None):
        """
        Ajoute un arbre à la forêt, avec un poids donné. Peut l'entraîner si X et y sont fournis.

        Paramètres :
        - tree : instance de la classe Tree.
        - X : array-like, données d'entraînement (optionnel).
        - y : array-like, étiquettes d'entraînement (optionnel).
        - weight : float, poids attribué à l'arbre dans le modèle (par défaut : 0.1).
        - forced_features : liste, caractéristiques à forcer dans l'entraînement (optionnel).
        """
        if X is not None and y is not None:
            tree.fit(X, y, forced_features)
            tree.classes_ = self.classes_

        self.trees.append(tree)
        self.weights.append(weight)
    
    def predict_proba(self, X):
        """
        Calcule les probabilités prédites pour chaque classe, moyennées par les poids des arbres.

        Paramètres :
        - X : array-like, données à prédire.

        Retour :
        - probas : ndarray de shape (n_samples, n_classes), probabilités finales pondérées.

        Exception :
        - ValueError si aucun arbre n'a été ajouté à la forêt.
        """
        if not self.trees:
            raise ValueError("Aucun arbre n'a été ajouté à la forêt.")

        total_weight = sum(self.weights)
        probas = np.zeros((X.shape[0], len(self.classes_)))
        self.per_tree_preds = []

        for tree, weight in zip(self.trees, self.weights):
            local_proba = tree.predict_proba(X)
            aligned_proba = np.zeros((X.shape[0], len(self.classes_)))

            for i, cls in enumerate(tree.classes_):
                col_index = np.where(self.classes_ == cls)[0][0]
                aligned_proba[:, col_index] = local_proba[:, i]
        
            preds_tree = self.classes_[np.argmax(aligned_proba, axis=1)]
            self.per_tree_preds.append(preds_tree)

            probas += weight * aligned_proba

        return probas / total_weight

    def predict(self, X):
        """
        Prédit les classes pour chaque échantillon en prenant la classe avec la probabilité maximale.

        Paramètres :
        - X : array-like, données à prédire.

        Retour :
        - prédictions : ndarray contenant la classe prédite pour chaque échantillon.
        """
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
