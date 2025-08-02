import numpy as np
import random
import math
from TreeNode import TreeNode

class Tree:
    def __init__(self, max_depth=3, min_samples_split=2):
        """
        Initialise un arbre de décision.

        Paramètres :
        - max_depth (int) : profondeur maximale de l'arbre.
        - min_samples_split (int) : nombre minimal d'échantillons requis pour effectuer un split.

        Attributs :
        - max_depth : profondeur maximale de l'arbre
        - min_samples_split : nombre minimal d'échantillons requis pour effectuer une division
        - root (TreeNode) : racine de l'arbre (None avant entraînement).
        - specific_splits (dict) : splits forcés sur certaines features (optionnel).
        - selected_features (list) : features utilisées pour construire l'arbre.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.specific_splits = {}  
        self.selected_features = None

    def fit(self, X, y, forced_features=None):
        """
        Entraîne l'arbre sur les données d'entrée.

        Paramètres :
        - X (array-like) : matrice des features (n_samples x n_features).
        - y (array-like) : vecteur des étiquettes.
        - forced_features (list ou None) : indices des features à forcer pour le split.

        Effets :
        - Remplit self.root avec la structure de l'arbre construite.
        """
        self.classes_ = np.unique(y)
        n_features = X.shape[1]

        if forced_features is not None:
            self.selected_features = forced_features
        else:
            n_to_select =  max(1, int(math.sqrt(n_features)))
            self.selected_features = random.sample(range(n_features), n_to_select)

        self.root = self._build_tree(
            np.array(X),
            np.array(y),
            available_features=self.selected_features
        )

    def _build_tree(self, X, y, depth=0, available_features=None):
        """
        Construit récursivement l'arbre de décision.

        Paramètres :
        - X (np.ndarray) : sous-ensemble des features.
        - y (np.ndarray) : sous-ensemble des labels.
        - depth (int) : profondeur actuelle dans l'arbre.
        - available_features (list) : features encore disponibles pour le split.

        Retour :
        - TreeNode : un noeud (interne ou feuille).
        """
        n_samples, n_features = X.shape
        num_classes = len(np.unique(y))

        if depth >= self.max_depth or n_samples < self.min_samples_split or num_classes == 1:
            leaf_value = self._majority_class(y)
            return TreeNode(value=leaf_value)

        best_feat, best_thresh = self._best_split(X, y, available_features)

        if best_feat is None:
            return TreeNode(value=self._majority_class(y))

        indices_left = X[:, best_feat] < best_thresh

        # Construction récursive des sous-arbres
        new_features = [f for f in available_features if f != best_feat]
        left = self._build_tree(X[indices_left], y[indices_left], depth + 1, new_features)
        right = self._build_tree(X[~indices_left], y[~indices_left], depth + 1, new_features)

        return TreeNode(best_feat, best_thresh, left, right)


    def _best_split(self, X, y, features):
        """
        Détermine le meilleur split en fonction de l'indice de Gini.

        Paramètres :
        - X (np.ndarray) : matrice des features.
        - y (np.ndarray) : vecteur des classes.
        - features (list) : indices des features à considérer.

        Retour :
        - (int ou None, float ou None) : meilleur feature et seuil associé.
        """
        best_gini = 1.0
        best_feat = None
        best_thresh = None

        for feature_index in features:
            if feature_index in self.specific_splits:
                thresholds = [self.specific_splits[feature_index]]
            else:
                thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                gini = self._gini_index(X, y, feature_index, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feat = feature_index
                    best_thresh = threshold

        return best_feat, best_thresh

    def _gini_index(self, X, y, feature_index, threshold):
        """
        Calcule l'indice de Gini pour un split donné.

        Paramètres :
        - X (np.ndarray) : matrice des features.
        - y (np.ndarray) : étiquettes associées.
        - feature_index (int) : indice de la feature considérée.
        - threshold (float) : seuil de split.

        Retour :
        - float : valeur de l'indice de Gini après le split.
        """
        left_mask = X[:, feature_index] < threshold
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 1.0  # Mauvais split

        def gini(y_subset):
            classes, counts = np.unique(y_subset, return_counts=True)
            probs = counts / counts.sum()
            return 1.0 - np.sum(probs ** 2)

        left_gini = gini(y[left_mask])
        right_gini = gini(y[right_mask])
        left_ratio = np.sum(left_mask) / len(y)
        right_ratio = 1.0 - left_ratio

        return left_ratio * left_gini + right_ratio * right_gini


    def _majority_class(self, y):
        """
        Renvoie la classe majoritaire parmi les labels fournis.

        Paramètres :
        - y (np.ndarray) : vecteur des classes.

        Retour :
        - int ou str : classe la plus fréquente.
        """
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]


    def predict(self, X):
        """
        Prédit la classe de chaque échantillon dans X.

        Paramètres :
        - X (array-like) : matrice des features (n_samples x n_features).

        Retour :
        - list : classes prédites pour chaque échantillon.
        """
        X = np.array(X)
        return [self._predict(inputs, self.root) for inputs in X]


    def _predict(self, x, node):
        """
        Prédit la classe d'un échantillon unique en parcourant l'arbre.

        Paramètres :
        - x (np.ndarray) : vecteur de features pour un échantillon.
        - node (TreeNode) : nœud courant dans l'arbre.

        Retour :
        - int ou str : classe prédite.
        """
        if node.value is not None:
            return node.value
        if x[node.feature_index] < node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)

    def predict_proba(self, X):
        """
        Donne la distribution des probabilités de classes pour chaque échantillon.

        Paramètres :
        - X (array-like) : matrice des features.

        Retour :
        - np.ndarray : tableau (n_samples x n_classes) avec les probabilités.
        """
        X = np.array(X)
        result = []
        for x in X:
            probs = self._predict_proba(x, self.root)
            result.append(probs)
        return np.array(result)

    def _predict_proba(self, x, node):
        """
        Calcule récursivement les probabilités de classe pour un échantillon.

        Paramètres :
        - x (np.ndarray) : vecteur de features.
        - node (TreeNode) : nœud courant dans l'arbre.

        Retour :
        - np.ndarray : vecteur de probabilités (longueur = nb classes).
        """
        if node.value is not None:
            # Une feuille : retourne un vecteur avec 100% pour la classe majoritaire
            probs = np.zeros(len(self.classes_))
            class_index = np.where(self.classes_ == node.value)[0][0]
            probs[class_index] = 1.0
            return probs
        if x[node.feature_index] < node.threshold:
            return self._predict_proba(x, node.left)
        else:
            return self._predict_proba(x, node.right)


    def print_tree(self, node=None, depth=0):
        """
        Affiche l’arbre de décision de façon textuelle.

        Paramètres :
        - node (TreeNode ou None) : nœud à afficher (défaut : racine).
        - depth (int) : niveau de profondeur pour l’indentation.
        """
        if node is None:
            node = self.root
        if node.value is not None:
            print(f"{'  ' * depth}Leaf: {node.value}")
        else:
            print(f"{'  ' * depth}X[{node.feature_index}] < {node.threshold}")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)

