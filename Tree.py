import numpy as np
from TreeNode import TreeNode

class Tree:
    def __init__(self, max_depth=3, min_samples_split=2):
        """
        Initialise un arbre de décision.

        - max_depth : profondeur maximale de l'arbre
        - min_samples_split : nombre minimal d'échantillons requis pour effectuer une division
        - self.root : racine de l'arbre, initialement vide
        - self.specific_splits : dictionnaire facultatif pour forcer des splits sur des features données
        - self.recalculation_nodes : dictionnaire permettant de retenir les nœuds à recalculer
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.specific_splits = {}  # ex: {0: 1.5, 2: 3.2} => impose des seuils sur certaines features
        # self.recalculation_nodes = {}  # ex: {'path': node} pour recalculer un sous-arbre
        # self.nodes_by_feature = {}  # ex: {feature_index: [TreeNode, TreeNode, ...]}

    def fit(self, X, y):
        """
        Point de départ : construction de l'arbre depuis la racine.
        - X : matrice d'entrée des features
        - y : vecteur des classes
        """
        self.classes_ = np.unique(y)
        n_features = np.array(X).shape[1]
        self.root = self._build_tree(np.array(X), np.array(y), available_features=list(range(n_features)))

    def _build_tree(self, X, y, depth=0, available_features=None):
        """
        Fonction récursive pour construire l'arbre.
        - Si la profondeur maximale est atteinte, ou si le nombre d'échantillons est trop faible,
        ou s'il n'y a qu'une seule classe, on crée une feuille.
        - Sinon, on cherche le meilleur split (en prenant en compte les splits spécifiques) 
        et on construit les sous-arbres gauche et droit.
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

        # # Sauvegarde du nœud pour recalcul éventuel
        # self.recalculation_nodes[f'depth_{depth}_feature_{best_feat}'] = (X, y)

        return TreeNode(best_feat, best_thresh, left, right)


    def _best_split(self, X, y, features):
        """
        Trouve le meilleur split en évaluant toutes les features :
        - Si une feature est dans specific_splits, on ne teste que le seuil imposé.
        - Sinon, on teste tous les seuils uniques possibles pour cette feature.
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
        Calcule l'indice de Gini après une séparation donnée (feature, seuil).
        Sert à mesurer la pureté des sous-groupes.
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
        Retourne la classe la plus fréquente dans y (majorité),
        utilisée pour les feuilles.
        """
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]


    def predict(self, X):
        """
        Prédit la classe pour chaque échantillon X.
        """
        X = np.array(X)
        return [self._predict(inputs, self.root) for inputs in X]


    def _predict(self, x, node):
        """
        Parcours récursif de l'arbre pour une seule ligne x
        jusqu’à atteindre une feuille.
        """
        if node.value is not None:
            return node.value
        if x[node.feature_index] < node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)

    def predict_proba(self, X):
        """
        Retourne la distribution des probabilités pour chaque classe.
        """
        X = np.array(X)
        result = []
        for x in X:
            probs = self._predict_proba(x, self.root)
            result.append(probs)
        return np.array(result)

    def _predict_proba(self, x, node):
        """
        Renvoie un vecteur de probabilités des classes à partir du nœud donné.
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
        Affiche récursivement l’arbre en format textuel.
        """
        if node is None:
            node = self.root
        if node.value is not None:
            print(f"{'  ' * depth}Leaf: {node.value}")
        else:
            print(f"{'  ' * depth}X[{node.feature_index}] < {node.threshold}")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)

